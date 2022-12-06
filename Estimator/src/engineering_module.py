import logging
import math
import statistics
import multiprocessing as mp
from multiprocessing import Process, Pool
import numpy as np

CARD_NUMBER = "card_number"
UNDER_SCORE = "_"


class Engineer:
    def __init__(self, time_base_field_name, encoded_field_names, feature_engineering_parameters):
        self.logger = logging.getLogger("estimator.server.database.engineer")
        self.time_base_field_name = time_base_field_name
        self.encoded_field_names = encoded_field_names
        self.feature_engineering_parameters = feature_engineering_parameters
        self.chosen_feature_field_name = self.feature_engineering_parameters.get("chosen_feature_field")
        self.intervals = self.feature_engineering_parameters.get("intervals")

    def process(self, encoded_input_data, previous_feature_and_time_base_array_by_intervals,
                index_of_feature_to_be_engineered, index_of_time_base_field):
        current_feature_value_to_be_engineered = encoded_input_data[index_of_feature_to_be_engineered]
        current_time_stamp_value = encoded_input_data[index_of_time_base_field]
        for interval, previous_feature_and_time_base_array in previous_feature_and_time_base_array_by_intervals.items():
            previous_feature_array = previous_feature_and_time_base_array[:, :1]
            previous_time_stamp_array = previous_feature_and_time_base_array[:, 1:]
            previous_feature_list = previous_feature_array.flatten().tolist()
            previous_time_stamp_list = previous_time_stamp_array.flatten().tolist()
            combined_feature_list = list()
            combined_time_stamp_list = list()
            combined_feature_list.append(current_feature_value_to_be_engineered)
            combined_feature_list.extend(previous_feature_list)
            combined_time_stamp_list.append(current_time_stamp_value)
            combined_time_stamp_list.extend(previous_time_stamp_list)
            generated_features_based_on_chosen_feature = self.get_generated_features_based_on_chosen_feature(
                current_feature_value_to_be_engineered, combined_feature_list)
            generated_features_based_on_transaction_number = self.get_generated_features_based_on_transaction_number(
                current_time_stamp_value, combined_time_stamp_list, interval)
            generated_features_based_on_transaction_interval = self.get_generated_features_based_on_transaction_interval(
                combined_time_stamp_list)
            encoded_input_data.extend(generated_features_based_on_chosen_feature)
            encoded_input_data.extend(generated_features_based_on_transaction_number)
            encoded_input_data.extend(generated_features_based_on_transaction_interval)
        return encoded_input_data

    def get_generated_features_based_on_chosen_feature(self, current_feature_value,
                                                       combined_feature_list):
        average_feature_value = statistics.mean(combined_feature_list)
        deviation_feature_value = statistics.stdev(combined_feature_list)
        median_feature_value = statistics.median(combined_feature_list)
        result = list()
        feature_per_average_feature_value = current_feature_value / average_feature_value if average_feature_value != 0 else 0
        result.append(feature_per_average_feature_value)
        feature_minus_average_feature_value = current_feature_value - average_feature_value
        result.append(feature_minus_average_feature_value)
        feature_per_median_feature_value = current_feature_value / median_feature_value if median_feature_value != 0 else 0
        result.append(feature_per_median_feature_value)
        feature_minus_median_feature_value = current_feature_value - median_feature_value
        result.append(feature_minus_median_feature_value)
        feature_minus_average_feature_per_deviation_feature_value = (current_feature_value - average_feature_value) / deviation_feature_value if deviation_feature_value != 0 else 0
        result.append(feature_minus_average_feature_per_deviation_feature_value)
        feature_minus_average_feature_minus_deviation_feature_value = current_feature_value - average_feature_value - deviation_feature_value
        result.append(feature_minus_average_feature_minus_deviation_feature_value)
        return result

    def get_generated_features_based_on_transaction_number(self, current_time_stamp_value, combined_time_stamp_list,
                                                           interval):
        transaction_list_on_last_day = list()
        for time_stamp in combined_time_stamp_list:
            if time_stamp > current_time_stamp_value - 1:
                transaction_list_on_last_day.append(time_stamp)
        transaction_number_on_last_day = len(transaction_list_on_last_day)
        average_transaction_number = len(combined_time_stamp_list) / interval
        median_transaction_number = self.get_median_transaction_number(combined_time_stamp_list)
        result = list()
        transaction_number_on_current_day_per_daily_average_transaction_number = transaction_number_on_last_day / average_transaction_number if average_transaction_number != 0 else 0
        result.append(transaction_number_on_current_day_per_daily_average_transaction_number)
        transaction_number_on_current_day_minus_daily_average_transaction_number = transaction_number_on_last_day - average_transaction_number
        result.append(transaction_number_on_current_day_minus_daily_average_transaction_number)
        transaction_number_on_current_day_per_daily_median_transaction_number = transaction_number_on_last_day / median_transaction_number if median_transaction_number != 0 else 0
        result.append(transaction_number_on_current_day_per_daily_median_transaction_number)
        transaction_number_on_current_day_minus_daily_median_transaction_number = transaction_number_on_last_day - median_transaction_number
        result.append(transaction_number_on_current_day_minus_daily_median_transaction_number)
        return result

    def get_median_transaction_number(self,combined_time_stamp_list):
        time_stamp_by_date_in_julian_format=dict()
        for time_stamp_date_in_julian_format in combined_time_stamp_list:
            current_day_in_julian_format=int(time_stamp_date_in_julian_format)
            if time_stamp_by_date_in_julian_format.get(current_day_in_julian_format) is None:
                time_stamps_on_given_day=list()
                time_stamps_on_given_day.append(time_stamp_date_in_julian_format)
                time_stamp_by_date_in_julian_format[current_day_in_julian_format]=time_stamps_on_given_day
            else:
                time_stamps_on_given_day=time_stamp_by_date_in_julian_format.get(current_day_in_julian_format)
                time_stamps_on_given_day.append(time_stamp_date_in_julian_format)
        transaction_numbers_on_days=list()
        for time_stamps in time_stamp_by_date_in_julian_format.values():
            transaction_numbers_on_days.append(len(time_stamps))
        median_transsction_number=statistics.median(transaction_numbers_on_days)
        return median_transsction_number

    def get_generated_features_based_on_transaction_interval(self, combined_time_stamp_list):
        number_of_time_stamps = len(combined_time_stamp_list)
        if number_of_time_stamps>=2:
            current_distance = combined_time_stamp_list[0] - combined_time_stamp_list[1]
            distances = list()
            for i in range(number_of_time_stamps - 1):
                distance = combined_time_stamp_list[i] - combined_time_stamp_list[i + 1]
                distances.append(distance)

            average_distance_between_transactions = statistics.mean(distances)
            median_distance_between_transactions = statistics.median(distances)
            result = list()
            average_distance_per_current_distance = average_distance_between_transactions / current_distance if current_distance != 0 else 0
            average_distance_time_minus_current_distance_between_transactions = average_distance_between_transactions - current_distance
            median_distance_per_current_distance_between_transactions = median_distance_between_transactions / current_distance if current_distance != 0 else 0
            median_distance_minus_current_distance_between_transactions = median_distance_between_transactions - current_distance
        else:
            average_distance_per_current_distance = 0
            average_distance_time_minus_current_distance_between_transactions = 0
            median_distance_per_current_distance_between_transactions = 0
            median_distance_minus_current_distance_between_transactions = 0

        result.append(average_distance_per_current_distance)
        result.append(average_distance_time_minus_current_distance_between_transactions)
        result.append(median_distance_per_current_distance_between_transactions)
        result.append(median_distance_minus_current_distance_between_transactions)
        return result
