import os
import h5py
import pickle
import powerlaw
import numpy as np
from tqdm import tqdm
from constants import *
from scipy.signal import find_peaks
from numba import jit, prange, types
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def load_failure_data(filenames, directory, loading_segments, verbose=False):
    """
    Load the failure data from hdf5 files
    """
    data_directory_name = directory
    hdf5_files = [filename for filename in filenames]

    if verbose:
        print(f'Loading {len(hdf5_files)} hdf5 files...')
        for filename in hdf5_files:
            print(f'- {filename}')

    all_failure_times = []
    all_failure_idx = []
    t0_s_all = []
    for hdf5_file in tqdm(hdf5_files):
        path_hdf5 = os.path.join(data_directory_name, hdf5_file)
        with h5py.File(path_hdf5, mode='r') as file:
            all_failure_times.append(file['Thermal/avalanches/t'][:loading_segments].flatten())
            all_failure_idx.append(file['Thermal/avalanches/idx'][:loading_segments].flatten())
            t0_s_all.append(file['Thermal/avalanches/t0'][0])
    if verbose:
        print(f'Finished loading failure times')
    for i in range(len(filenames)):
        all_failure_times[i] = all_failure_times[i] - t0_s_all[i]
    return all_failure_times, all_failure_idx


# @jit(nopython=True, parallel=True)
# def count_events_after(failure_times, t_events, time_windows):
#     n_events_after = np.zeros((len(t_events), len(time_windows)))
#     for i in prange(len(t_events)):
#         t_event = t_events[i]
#         for j in range(len(time_windows)):
#             window = time_windows[j]
#             t_end = t_event + window
#             count = 0
#             for k in range(len(failure_times)):
#                 if t_event <= failure_times[k] <= t_end:
#                     count += 1
#             n_events_after[i, j] = count
#     return n_events_after


@jit(nopython=True, parallel=True)
def count_events_after(failure_times, t_events, time_windows):
    n_events_after = np.zeros((len(t_events), len(time_windows)))
    cumulative_counts = np.arange(1, len(failure_times) + 1)
    for i in prange(len(t_events)):
        t_event = t_events[i]
        event_index = np.searchsorted(failure_times, t_event, side='left')
        for j in prange(len(time_windows)):
            window = time_windows[j]
            t_end = t_event + window
            end_index = np.searchsorted(failure_times, t_end, side='right')
            count = cumulative_counts[end_index - 1] - (cumulative_counts[event_index - 1] if event_index > 0 else 0)
            n_events_after[i, j] = count
    return n_events_after


@jit(nopython=True, parallel=True)
def count_events_before(failure_times, t_events, time_windows):
    n_events_before = np.zeros((len(t_events), len(time_windows)))
    cumulative_counts = np.arange(1, len(failure_times) + 1)
    for i in prange(len(t_events)):
        t_event = t_events[i]
        event_index = np.searchsorted(failure_times, t_event, side='right') - 1
        for j in prange(len(time_windows)):
            window = time_windows[j]
            t_start = t_event - window
            start_index = np.searchsorted(failure_times, t_start, side='left')
            count = cumulative_counts[event_index] - (cumulative_counts[start_index - 1] if start_index > 0 else 0)
            n_events_before[i, j] = count
    return n_events_before


# @jit(nopython=True, parallel=True)
# def count_events_before(failure_times, t_events, time_windows):
#     n_events_before = np.zeros((len(t_events), len(time_windows)))
#     for i in prange(len(t_events)):
#         t_event = t_events[i]
#         for j in range(len(time_windows)):
#             window = time_windows[j]
#             t_start = t_event - window
#             count = 0
#             for k in range(len(failure_times)):
#                 if t_start <= failure_times[k] <= t_event:
#                     count += 1
#             n_events_before[i, j] = count
#     return n_events_before


def sample_filter_peaks(peak_times, peak_heights, heights, lower_bounds, upper_bounds):
    filtered_peak_t = {}
    filtered_peak_h = {}
    for h, lower_bound, upper_bound in zip(heights, lower_bounds, upper_bounds):  # loop over ranges
        filtered_peak_indices = np.where((lower_bound <= peak_heights) & (peak_heights <= upper_bound))[0]
        if len(filtered_peak_indices) == 0:
            continue
        filtered_peak_t[h] = peak_times[filtered_peak_indices]
        filtered_peak_h[h] = peak_heights[filtered_peak_indices]
    return filtered_peak_t, filtered_peak_h


def sample_heights_stats(times, sigma, threshold):
    cumulative = np.arange(1, len(times) + 1)
    step_function = interp1d(times, cumulative, kind='previous', fill_value="extrapolate")
    t_interp = np.linspace(times[0], times[-1], num=len(times))
    cumulative_interp = step_function(t_interp)

    # Compute the activity and it's gaussian convolution
    activity = np.gradient(cumulative_interp, t_interp)
    smooth_activity = gaussian_filter1d(activity, sigma=sigma)

    # Find peaks
    peak_idx, _ = find_peaks(smooth_activity, threshold=threshold)
    peak_times_neighbours = t_interp[peak_idx]
    peak_times = np.array([times[np.argmin(np.abs(times - t))] for t in peak_times_neighbours])

    peak_heights = smooth_activity[peak_idx]

    return {
        "t_interp": t_interp,
        "smooth_activity": smooth_activity,
        "peak_idx": peak_idx,
        "peak_times": peak_times,
        "peak_heights": peak_heights,
        "mean_peak_height": np.mean(peak_heights),
        "std_peak_height": np.std(peak_heights, ddof=1)
    }


def all_sample_heights_stats(all_failure_times, sigma, threshold, save=False, filename=None):
    all_heights_stats = {"all_peak_h": [],
                         "mean_all_h": [],
                         "std_all_h": [],
                         "n_peaks": 0}
    for i in tqdm(range(len(all_failure_times)), total=len(all_failure_times), desc="Computing heights"):
        heights_stats = sample_heights_stats(all_failure_times[i], sigma, threshold)
        all_heights_stats["all_peak_h"].extend(heights_stats["peak_heights"])

    all_heights_stats["mean_all_h"] = np.mean(all_heights_stats["all_peak_h"])
    all_heights_stats["std_all_h"] = np.std(all_heights_stats["all_peak_h"], ddof=1)
    all_heights_stats["n_peaks"] = len(all_heights_stats["all_peak_h"])
    if save:
        height_dir = os.path.join(SAVE_DIR, "heights")
        if not os.path.exists(height_dir):
            os.makedirs(height_dir)
        save_path = os.path.join(height_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(all_heights_stats, f)
    return all_heights_stats


def concatenate_result(result_dict, heights, n_samples):
    all_result = {}
    for h in heights:
        all_result[h] = np.concatenate([result_dict[i][h] for i in range(n_samples)], axis=0)
    return all_result


def sample_analysis(times,
                    sigma,
                    threshold,
                    compute_activity_for_random_heights=False,
                    n_random_peaks=10,
                    compute_activity_for_given_heights=False,
                    heights=None,
                    var_h=0.05,
                    t_start=1e-7,
                    t_stop=2,
                    num_time_windows=30,
                    verbose=False,
                    save=False,
                    filename=None):

    """
    Perform the sample analysis for the given time series of failures.

    :param times: times of all the failures
    :param sigma: standard deviation of the gaussian filter (defines a local time scale)
    :param threshold: threshold for finding peaks in the smoothed activity
    :param compute_activity_for_random_heights: whether to compute activity for random heights
    :param n_random_peaks: number of random peaks
    :param compute_activity_for_given_heights: whether to compute activity for given heights
    :param heights: heights for which to compute the activity
    :param var_h: variance of the peak heights (% of the height)
    :param t_start: start time for the time windows
    :param t_stop: stop time for the time windows
    :param num_time_windows: number of time windows to consider when counting events
    :param verbose: whether to print the results
    :param save: whether to save the results
    :param filename: name of the file to save the results

    :return output_dict: dictionary containing the results of the analysis
    """
    if heights is None:
        heights = [0]
    output_dict = {}

    heights_stats = sample_heights_stats(times, sigma, threshold)
    t_interp = heights_stats["t_interp"]
    smooth_activity = heights_stats["smooth_activity"]
    peak_times = heights_stats["peak_times"]
    peak_heights = heights_stats["peak_heights"]
    peak_idx = heights_stats["peak_idx"]

    if verbose:
        print(f"Number of peaks = {len(peak_heights)}")

    fit = powerlaw.Fit(peak_heights, xmin=min(peak_heights), xmax=max(peak_heights), discrete=False)
    exp = fit.power_law.alpha
    err = fit.power_law.sigma
    if verbose:
        print(f"The power law exponent of the distribution of peak heights is = -{exp} +/- {err}")

    if compute_activity_for_random_heights:
        print("Computing activity for random heights")

        np.random.seed(0)
        random_peak_idx = np.random.choice(peak_idx, size=n_random_peaks, replace=False)
        random_peak_idx.sort()
        random_peak_times = t_interp[random_peak_idx]
        random_peak_heights = smooth_activity[random_peak_idx]

        # Time windows to consider when counting events
        time_windows = np.geomspace(t_start, t_stop, num=num_time_windows)

        # Keep times within the range
        peak_cond = (random_peak_times-t_stop > 0) & (random_peak_times+t_stop < times[-1])
        random_peak_idx = random_peak_idx[peak_cond]
        random_peak_times = random_peak_times[peak_cond]
        random_peak_heights = random_peak_heights[peak_cond]
        print("- Number of random peaks = ", len(random_peak_times))

        # Count events before and after the peaks
        random_n_before = count_events_before(times, random_peak_times, time_windows)
        random_mean_n_before = np.mean(random_n_before, axis=0)
        random_n_after = count_events_after(times, random_peak_times, time_windows)
        random_mean_n_after = np.mean(random_n_after, axis=0)

        # Compute the activity before and after the peaks
        random_mean_a_before = random_mean_n_before / time_windows
        random_mean_a_after = random_mean_n_after / time_windows

        # Save the results
        output_dict["random_time_windows"] = time_windows
        output_dict["random_n_before"] = random_n_before
        output_dict["random_n_after"] = random_n_after
        output_dict["random_mean_n_before"] = random_mean_n_before
        output_dict["random_mean_n_after"] = random_mean_n_after
        output_dict["random_mean_a_before"] = random_mean_a_before
        output_dict["random_mean_a_after"] = random_mean_a_after
        output_dict["n_random_peaks"] = len(random_peak_times)

        if save:
            random_peaks_dir = os.path.join(SAVE_DIR, "random_peaks")
            if not os.path.exists(random_peaks_dir):
                os.makedirs(random_peaks_dir)
            save_path = os.path.join(random_peaks_dir, filename)
            with open(save_path, 'wb') as f:
                pickle.dump(output_dict, f)

    if compute_activity_for_given_heights:
        print("Computing activity for given heights")

        # Filter peaks based on height
        DELTA_H = var_h * heights
        LOWER_BOUNDS = heights - DELTA_H
        UPPER_BOUNDS = heights + DELTA_H
        time_windows = np.geomspace(t_start, t_stop, num=num_time_windows)

        # Filter peaks by the given heights
        filtered_peak_times, filtered_peak_heights = sample_filter_peaks(peak_times=peak_times,
                                                                         peak_heights=peak_heights,
                                                                         heights=heights,
                                                                         lower_bounds=LOWER_BOUNDS,
                                                                         upper_bounds=UPPER_BOUNDS)

        # Filter valid times
        n_peaks_before ={}
        n_peaks_after = {}
        for h in heights:
            peak_cond = (filtered_peak_times[h] - t_stop > 0) & (filtered_peak_times[h] + t_stop < times[-1])
            filtered_peak_times[h] = filtered_peak_times[h][peak_cond]
            filtered_peak_heights[h] = filtered_peak_heights[h][peak_cond]
            n_peaks_before[h] = len(filtered_peak_times[h])
            n_peaks_after[h] = len(filtered_peak_times[h])
        print("- Number of peaks before = ", n_peaks_before)
        print("- Number of peaks after = ", n_peaks_after)

        # Number of events before and after the peaks
        n_before = {}
        mean_n_before = {}
        for h in tqdm(heights, total=len(heights), desc="Computing events before peaks"):
            n_before[h] = count_events_before(times, filtered_peak_times[h], time_windows)
            mean_n_before[h] = np.mean(n_before[h], axis=0)
        n_after = {}
        mean_n_after = {}
        for h in tqdm(heights, total=len(heights), desc="Computing events after peaks"):
            n_after[h] = count_events_after(times, filtered_peak_times[h], time_windows)
            mean_n_after[h] = np.mean(n_after[h], axis=0)

        # Mean activity before and after peaks for each height
        mean_a_before = {}
        mean_a_after = {}
        for h in heights:
            mean_a_before[h] = mean_n_before[h] / time_windows
            mean_a_after[h] = mean_n_after[h] / time_windows

        # Normalize the activity curves by the first value
        norm_mean_a_before = {}
        norm_mean_a_after = {}
        for h in heights:
            if mean_a_before[h][0] == 0:
                norm_mean_a_before[h] = np.zeros_like(mean_a_before[h])
            else:
                norm_mean_a_before[h] = mean_a_before[h] / mean_a_before[h][0]

            if mean_a_after[h][0] == 0:
                norm_mean_a_after[h] = np.zeros_like(mean_a_after[h])
            else:
                norm_mean_a_after[h] = mean_a_after[h] / mean_a_after[h][0]

        # Mean of the normalized activity curves
        mean_norm_mean_a_before = np.mean([norm_mean_a_before[h] for h in heights], axis=0)
        mean_norm_mean_a_after = np.mean([norm_mean_a_after[h] for h in heights], axis=0)

        # Save the results
        output_dict["time_windows"] = time_windows
        output_dict["n_before"] = n_before
        output_dict["n_after"] = n_after
        output_dict["mean_n_before"] = mean_n_before
        output_dict["mean_n_after"] = mean_n_after
        output_dict["mean_a_before"] = mean_a_before
        output_dict["mean_a_after"] = mean_a_after
        output_dict["norm_mean_a_before"] = norm_mean_a_before
        output_dict["norm_mean_a_after"] = norm_mean_a_after
        output_dict["mean_norm_mean_a_before"] = mean_norm_mean_a_before
        output_dict["mean_norm_mean_a_after"] = mean_norm_mean_a_after
        output_dict["n_peaks"] = n_peaks_before
    output_dict["peak_times"] = peak_times
    output_dict["peak_heights"] = peak_heights

    if save:
        sample_data_dir = os.path.join(SAVE_DIR, "sample_data")
        if not os.path.exists(sample_data_dir):
            os.makedirs(sample_data_dir)
        save_path = os.path.join(sample_data_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(output_dict, f)

    return output_dict


def n_events_around_microscopic_event(all_failure_times, N_random_peaks=20, t_start=1e-5, t_stop=1, num_time_windows=1000, save=False, filename=None):
    time_windows = np.geomspace(t_start, t_stop, num=num_time_windows)
    all_n_events_before = []
    all_n_events_after = []
    for failure_times in tqdm(all_failure_times, total=len(all_failure_times), desc="Computing events around microscopic events"):
        t_events = np.random.choice(failure_times, N_random_peaks, replace=False)

        valid_events = (t_events - t_stop > 0) & (t_events + t_stop < failure_times[-1])

        n_events_before = count_events_before(failure_times, t_events[valid_events], time_windows)
        n_events_after = count_events_after(failure_times, t_events[valid_events], time_windows)
        all_n_events_before.extend(n_events_before)
        all_n_events_after.extend(n_events_after)

    mean_n_before = np.mean(all_n_events_before, axis=0)
    mean_n_after = np.mean(all_n_events_after, axis=0)
    std_n_before = np.std(all_n_events_before, axis=0, ddof=1)
    std_n_after = np.std(all_n_events_after, axis=0, ddof=1)

    output_dict = {
        "time_windows": time_windows,
        "n_before": all_n_events_before,
        "n_after": all_n_events_after,
        "mean_n_before": mean_n_before,
        "mean_n_after": mean_n_after,
        "std_n_before": std_n_before,
        "std_n_after": std_n_after
    }

    if save:
        sample_data_dir = os.path.join(SAVE_DIR, "random_microscopic_events")
        if not os.path.exists(sample_data_dir):
            os.makedirs(sample_data_dir)
        save_path = os.path.join(sample_data_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(output_dict, f)

    return output_dict