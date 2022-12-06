import logging
import math
import statistics
import multiprocessing as mp
from multiprocessing import Process, Pool
import numpy as np

CARD_NUMBER = "card_number"
UNDER_SCORE = "_"


class Engineer:
    def __init__(self, time_base_field_name, encoded_field_names, fraud_type_field_name,feature_engineering_parameters, used_cpu_core):
        self.logger = logging.getLogger("train.server.database.engineer")
        self.time_base_field_name = time_base_field_name
        self.encoded_field_names = encoded_field_names
        self.fraud_type_field_name=fraud_type_field_name
        self.feature_engineering_parameters = feature_engineering_parameters
        self.chosen_feature_field_name = self.feature_engineering_parameters.get("chosen_feature_field")
        self.intervals = self.feature_engineering_parameters.get("intervals")
        self.used_cpu_core = used_cpu_core

    # 3.1.7
    def create_new_features(self, dataset):
        self.logger.info(f"Used CPU logical core number: {self.used_cpu_core}")
        feature_engineered_field_names = self.crete_feature_engineered_field_names(self.encoded_field_names,
                                                                                   self.chosen_feature_field_name,
                                                                                   self.intervals)
        transaction_by_card_number = self.get_transaction_by_card_number(dataset, self.encoded_field_names)
        input_collection = list()
        for card_number in transaction_by_card_number:
            input_collection.append(transaction_by_card_number.get(card_number))
        results_from_threads = list()
        with Pool(processes=self.used_cpu_core) as pool:
            results_from_threads = pool.map(self.process_dataset_of_single_cardnumber, input_collection)
            self.logger.info(f"Number of sublists from threads: {len(results_from_threads)}")
        num_of_sub_results=len(results_from_threads)
        concatenated_results=np.empty([0,0])
        for i in range(num_of_sub_results):
            if i==0:
                concatenated_results=results_from_threads[i]
            else:
                concatenated_results=np.concatenate((concatenated_results,results_from_threads[i]),axis=0)

        return concatenated_results, feature_engineered_field_names

    def crete_feature_engineered_field_names(self, field_names, chosen_feature_field_name, time_intervals):
        result = list()
        real_fraud_field_name=None
        for field_name in field_names:
            if field_name==self.fraud_type_field_name:
                real_fraud_field_name=field_name
            else:
                result.append(field_name)
        for interval in time_intervals:
            result.append(chosen_feature_field_name + "_current_per_average_" + str(interval))
            result.append(chosen_feature_field_name + "_current_minus_average_" + str(interval))
            result.append(chosen_feature_field_name + "_current_per_median_" + str(interval))
            result.append(chosen_feature_field_name + "_current_minus_median_" + str(interval))
            result.append(
                chosen_feature_field_name + "_current_minus_average_" + str(interval) + "_per_deviation_" + str(
                    interval))
            result.append(
                chosen_feature_field_name + "_current_minus_average_" + str(interval) + "_minus_deviation_" + str(
                    interval))
            result.append("tr_nr_per_avg_nr_" + str(interval))
            result.append("tr_nr_minus_avg_nr_" + str(interval))
            result.append("tr_nr_per_median_nr_" + str(interval))
            result.append("tr_nr_minus_median_nr_" + str(interval))
            result.append("avg_tr_interval_per_tr_interval_" + str(interval))
            result.append("avg_tr_interval_minus_tr_interval_" + str(interval))
            result.append("median_tr_interval_per_tr_interval_" + str(interval))
            result.append("median_tr_interval_minus_tr_interval_" + str(interval))
        result.append(real_fraud_field_name)
        return result

    def get_transaction_by_card_number(self, dataset, encoded_field_names):
        result = dict()
        for record in dataset:
            index = encoded_field_names.index(CARD_NUMBER)
            card_number = record[index]
            if result.get(card_number) is None:
                transactions = list()
                transactions.append(record.tolist())
                result[card_number] = transactions
            else:
                transactions = result.get(card_number)
                transactions.append(record.tolist())
        return result

    def process_dataset_of_single_cardnumber(self, input_data_of_given_card_number):
        number_of_new_feature = len(self.intervals) * 14
        extended_dataset = list()
        data_set = np.array(input_data_of_given_card_number)
        transaction_features = data_set[:, :-1]
        transaction_labels = data_set[:, -1:]
        length = len(transaction_features)
        for i in range(length):
            transaction_feature = transaction_features[i]
            index_of_feature_to_be_engineered = self.encoded_field_names.index(self.chosen_feature_field_name)
            feature_value = transaction_feature[index_of_feature_to_be_engineered]
            index_of_time_stamp = self.encoded_field_names.index(self.time_base_field_name)
            current_time_stamp = transaction_feature[index_of_time_stamp]
            transaction_feature_list = list(transaction_feature)

            if i == length - 1:
                for j in range(number_of_new_feature):
                    transaction_feature_list.append(0)
                extended_dataset.append(transaction_feature_list)
            else:
                feature_to_be_engineered_by_time_stamp, time_stamps = self.get_feature_to_be_engineered_by_timestamp_before_given_time_stamp(
                    transaction_features, current_time_stamp, index_of_feature_to_be_engineered, index_of_time_stamp)

                generated_features = list()

                generated_features_based_on_chosen_feature = self.get_generated_features_based_on_chosen_feature(
                    current_time_stamp, feature_value, feature_to_be_engineered_by_time_stamp)

                generated_features_based_on_transaction_number = self.get_generated_features_based_on_transaction_number(
                    current_time_stamp, time_stamps)

                generated_features_based_on_transaction_interval = self.get_generated_features_based_on_transaction_interval(
                    current_time_stamp, time_stamps)

                generated_features.extend(generated_features_based_on_chosen_feature)
                generated_features.extend(generated_features_based_on_transaction_number)
                generated_features.extend(generated_features_based_on_transaction_interval)
                number_of_generated_features = len(generated_features)
                for interval in self.intervals:
                    for k in range(number_of_generated_features):
                        transaction_feature_list.append(generated_features[k].get(interval))

                label = transaction_labels[i][0]
                transaction_feature_list.append(label)
                transaction_feature_tuple = tuple(transaction_feature_list)
                extended_dataset.append(transaction_feature_tuple)
                result=np.array(extended_dataset)
        return result

    def get_feature_to_be_engineered_by_timestamp_before_given_time_stamp(self, data_set, given_time_stamp,
                                                                          index_of_feature_to_be_engineered,
                                                                          index_of_timestamp):
        feature_by_timestamp = dict()
        time_stamps = list()
        for record in data_set:
            current_time_stamp_value = record[index_of_timestamp]
            feature_to_be_engineered_value = record[index_of_feature_to_be_engineered]
            if current_time_stamp_value < given_time_stamp:
                time_stamps.append(current_time_stamp_value)
                feature_by_timestamp[current_time_stamp_value] = feature_to_be_engineered_value
        return feature_by_timestamp, time_stamps

    def get_generated_features_based_on_chosen_feature(self, current_timestamp, feature_value,
                                                       feature_to_be_engineered_by_timestamp):

        feature_values_by_retrospective_time = dict()
        for interval in self.intervals:
            feature_values = list()
            for timestamp_in_julian_date, feature_value in feature_to_be_engineered_by_timestamp.items():
                if timestamp_in_julian_date > current_timestamp - interval:
                    feature_values.append(feature_value)
            feature_values_by_retrospective_time[interval] = feature_values

        average_feature_value_by_retrospective_time = dict()
        deviation_feature_value_by_retrospective_time = dict()
        median_feature_value_by_retrospective_time = dict()

        for interval in self.intervals:
            feature_values = feature_values_by_retrospective_time.get(interval)
            try:
                average_feature_value_by_retrospective_time[interval] = statistics.mean(
                    feature_values) if feature_values else 0
                median_feature_value_by_retrospective_time[interval] = statistics.median(
                    feature_values) if feature_values else 0
                deviation_feature_value_by_retrospective_time[interval] = statistics.stdev(feature_values) if len(
                    feature_values) >= 2 else 0
            except TypeError as e:
                self.logger.error("error")

        feature_per_average_feature_value_by_retrospective_time = dict()
        feature_minus_average_feature_value_by_retrospective_time = dict()
        feature_per_median_feature_value_by_retrospective_time = dict()
        feature_minus_median_feature_value_by_retrospective_time = dict()
        feature_minus_average_feature_per_deviation_feature_value_by_retrospective_time = dict()
        feature_minus_average_feature_minus_deviation_feature_value_by_retrospective_time = dict()

        for interval in self.intervals:
            feature_per_average_feature_value_by_retrospective_time[
                interval] = feature_value / average_feature_value_by_retrospective_time.get(
                interval) if average_feature_value_by_retrospective_time.get(interval) != 0 else 0
            feature_minus_average_feature_value_by_retrospective_time[
                interval] = feature_value - average_feature_value_by_retrospective_time.get(interval)
            feature_per_median_feature_value_by_retrospective_time[
                interval] = feature_value / median_feature_value_by_retrospective_time.get(
                interval) if median_feature_value_by_retrospective_time.get(interval) != 0 else 0
            feature_minus_median_feature_value_by_retrospective_time[
                interval] = feature_value - median_feature_value_by_retrospective_time.get(interval)
            feature_minus_average_feature_per_deviation_feature_value_by_retrospective_time[
                interval] = feature_value / deviation_feature_value_by_retrospective_time.get(
                interval) if deviation_feature_value_by_retrospective_time.get(interval) != 0 else 0
            feature_minus_average_feature_minus_deviation_feature_value_by_retrospective_time[
                interval] = feature_value - deviation_feature_value_by_retrospective_time.get(interval)
        return [feature_per_average_feature_value_by_retrospective_time,
                feature_minus_average_feature_value_by_retrospective_time,
                feature_per_median_feature_value_by_retrospective_time,
                feature_minus_median_feature_value_by_retrospective_time,
                feature_minus_average_feature_per_deviation_feature_value_by_retrospective_time,
                feature_minus_average_feature_minus_deviation_feature_value_by_retrospective_time]

    def get_feature_values_by_julian_date(self, feature_and_timestamp_values):
        feature_values_by_date_dictionary = dict()
        for current_feature_and_timestamp in feature_and_timestamp_values:
            current_feature = current_feature_and_timestamp[0]
            timestamp = current_feature_and_timestamp[1]
            current_julian_date = int(timestamp)
            if feature_values_by_date_dictionary.get(current_julian_date) is None:
                feature_collection = list()
                feature_collection.append(current_feature)
                feature_values_by_date_dictionary[current_julian_date] = feature_collection
            else:
                feature_collection = feature_values_by_date_dictionary.get(current_julian_date)
                feature_collection.append(current_feature)
        return feature_values_by_date_dictionary

    def get_generated_features_based_on_transaction_number(self, current_time_stamp,
                                                           time_stamps):

        transaction_number_on_current_day = 1
        for time_stamp_in_julian_format in time_stamps:
            if time_stamp_in_julian_format > current_time_stamp - 1:
                transaction_number_on_current_day += 1

        transaction_number_by_interval = dict()
        for interval in self.intervals:
            transaction_number = 0
            transaction_number_by_day_order_number = dict()
            for i in range(interval):
                transaction_number = 0
                upper_boundara = current_time_stamp - i
                lower_boundary = current_time_stamp - (i + 1)
                for time_stamp_in_julian_format in time_stamps:
                    if time_stamp_in_julian_format < upper_boundara and time_stamp_in_julian_format > lower_boundary:
                        transaction_number += 1
                transaction_number_by_day_order_number[i] = transaction_number
            transactions_on_days = list()
            for transaction_number_on_given_day in transaction_number_by_day_order_number.values():
                transactions_on_days.append(transaction_number_on_given_day)
            transaction_number_by_interval[interval] = transactions_on_days

        daily_average_transaction_number_by_retrospective_time = dict()
        daily_median_transaction_number_by_retrospective_time = dict()

        for interval in self.intervals:
            transaction_number = transaction_number_by_interval.get(interval)
            daily_average_transaction_number_by_retrospective_time[interval] = statistics.mean(
                transaction_number) if transaction_number else 0
            daily_median_transaction_number_by_retrospective_time[interval] = statistics.median(
                transaction_number) if transaction_number else 0

        transaction_number_on_current_day_per_daily_average_transaction_number_by_retrospective_time = dict()
        transaction_number_on_current_day_minus_daily_average_transaction_number_by_retrospective_time = dict()
        transaction_number_on_current_day_per_daily_median_transaction_number_by_retrospective_time = dict()
        transaction_number_on_current_day_minus_daily_median_transaction_number_by_retrospective_time = dict()
        for interval in self.intervals:
            average_transaction_number = daily_average_transaction_number_by_retrospective_time.get(interval)
            median_transaction_number = daily_median_transaction_number_by_retrospective_time.get(interval)
            transaction_number_on_current_day_per_daily_average_transaction_number_by_retrospective_time[
                interval] = transaction_number_on_current_day / average_transaction_number if average_transaction_number != 0 else 0
            transaction_number_on_current_day_minus_daily_average_transaction_number_by_retrospective_time[
                interval] = transaction_number_on_current_day - average_transaction_number
            transaction_number_on_current_day_per_daily_median_transaction_number_by_retrospective_time[
                interval] = transaction_number_on_current_day / median_transaction_number if median_transaction_number != 0 else 0
            transaction_number_on_current_day_minus_daily_median_transaction_number_by_retrospective_time[
                interval] = transaction_number_on_current_day - median_transaction_number

        return [transaction_number_on_current_day_per_daily_average_transaction_number_by_retrospective_time,
                transaction_number_on_current_day_minus_daily_average_transaction_number_by_retrospective_time,
                transaction_number_on_current_day_per_daily_median_transaction_number_by_retrospective_time,
                transaction_number_on_current_day_minus_daily_median_transaction_number_by_retrospective_time]

    def get_generated_features_based_on_transaction_interval(self, current_timestamp,
                                                             time_stamps):
        current_distance_between_transactions = current_timestamp - time_stamps[0] if time_stamps else 0

        distances_by_retrospective_time = dict()
        length_of_timestamp_collection = len(time_stamps)
        for interval in self.intervals:
            distances = list()
            for i in range(0, length_of_timestamp_collection - 1, 1):
                next_time_stamp = time_stamps[i + 1]
                current_time_stamp = time_stamps[i]
                distance_between_transactions = current_time_stamp - next_time_stamp
                distances.append(distance_between_transactions)
            distances_by_retrospective_time[interval] = distances

        average_distance_between_transactions_by_retrospective_time = dict()
        median_distance_between_transactions_by_retrospective_time = dict()
        for interval in self.intervals:
            distances = distances_by_retrospective_time.get(interval)
            average_distance_between_transactions_by_retrospective_time[interval] = statistics.mean(
                distances) if distances else 0
            median_distance_between_transactions_by_retrospective_time[interval] = statistics.median(
                distances) if distances else 0

        average_distance_per_current_distance_between_transactions_by_retrospective_time = dict()
        average_distance_time_minus_current_distance_between_transactions_by_retrospective_time = dict()
        median_distance_per_current_distance_between_transactions_by_retrospective_time = dict()
        median_distance_minus_current_distance_between_transactions_by_retrospective_time = dict()

        for interval in self.intervals:
            average_distance = average_distance_between_transactions_by_retrospective_time.get(interval)
            median_distance = median_distance_between_transactions_by_retrospective_time.get(interval)
            average_distance_per_current_distance_between_transactions_by_retrospective_time[
                interval] = average_distance / current_distance_between_transactions if current_distance_between_transactions != 0 else 0
            average_distance_time_minus_current_distance_between_transactions_by_retrospective_time[
                interval] = average_distance - current_distance_between_transactions
            median_distance_per_current_distance_between_transactions_by_retrospective_time[
                interval] = median_distance / current_distance_between_transactions if current_distance_between_transactions != 0 else 0
            median_distance_minus_current_distance_between_transactions_by_retrospective_time[
                interval] = median_distance - current_distance_between_transactions

        return [average_distance_per_current_distance_between_transactions_by_retrospective_time,
                average_distance_time_minus_current_distance_between_transactions_by_retrospective_time,
                median_distance_per_current_distance_between_transactions_by_retrospective_time,
                median_distance_minus_current_distance_between_transactions_by_retrospective_time]
