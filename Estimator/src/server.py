import sys
from datetime import datetime
import time
import logging
import os
import pandas as pd
import pickle
from xgboost import XGBClassifier
from flask import Flask, jsonify, request, Response

import database_module
import engineering_module

FRAUD_TYPE = "fraud_type"
ID = "id"
TYPE = "type"
NAME = "name"
INT = "int"
FLOAT = "float"
STRING = "str"
DATETIME = "datetime"
LABEL_ENCODER = "label_encoder"
JULIAN = "julian"
INTERVALS = "intervals"
CARD_NUMBER = "card_number"
CHOSEN_FEATURE_FIELD_FOR_FEATURE_ENGINEERING = "chosen_feature_field"
COMMON_FRAUD_SCHEMA = "common_fraud"

server = Flask(__name__)


class InValidRequestParameterException(Exception):
    pass


class NotImplementedConversionTypeException(Exception):
    pass


class MissingInputFieldNameException(Exception):
    pass


@server.route('/prediction', methods=['POST'])
def get_prediction():
    if not request.is_json:
        error_message = "request body must be JSON"
        logger.error(error_message)
        return Response(error_message, status=415)
    request_parameter = request.get_json()
    try:
        start_predict = time.time()
        result = estimator_holder.predict_using_all_estimator(request_parameter)
        finish_predict = time.time()
        predict_processing_time = finish_predict - start_predict
        logger.info(f"prediction finished, result: {result}, processing time: {predict_processing_time}")
        return jsonify(result)
    except MissingInputFieldNameException as e:
        details = e.args[0]
        message = details.get("message")
        return Response(message, status=400)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message")
        return Response(message, status=400)


@server.route('/estimator', methods=['POST'])
def add_estimator():
    if not request.is_json:
        error_message = "request body must be JSON"
        logger.error(error_message)
        return Response(error_message, status=415)
    request_parameter = request.get_json()
    id = request_parameter.get("id")
    registered_estimator_ids = estimator_holder.get_all_registered_estimator_id()
    if id in registered_estimator_ids:
        message = f"estimator with this id is registered yet, id: {id}"
        logger.error(message)
        return Response(message, status=400)
    try:
        estimator_ids = estimator_holder.register_estimator(id)
        logger.info(f"estimator (id: {id}) registered")
        return jsonify(estimator_ids)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message")
        return Response(message, status=400)


@server.route('/estimator', methods=['GET'])
def get_estimator():
    result = estimator_holder.get_all_registered_estimator_id()
    return jsonify(result)


@server.route('/estimator/<id>', methods=['DELETE'])
def delete_estimator(id):
    try:
        estimator_ids = estimator_holder.delete_estimator(id)
        logger.info(f"estimator (id: {id}) deleted")
        return jsonify(estimator_ids)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message")
        logger.error(message)
        return Response(message, status=400)


class EstimatorHolder:
    def __init__(self):
        self.estimator_by_id = dict()

    def register_estimator(self, id):
        available_estimator_ids = database_handler.get_available_estimator_ids()
        if id not in available_estimator_ids:
            message = f"estimator doesn't exist, estimator id: {id}"
            logger.error(message)
            raise InValidRequestParameterException({"message": message})
        estimator = Estimator(id)
        estimator.initialize()
        self.estimator_by_id[id] = estimator
        return self.get_all_registered_estimator_id()

    def delete_estimator(self, id):
        try:
            self.estimator_by_id.pop(int(id))
        except KeyError:
            message = f"estimator isn't registered with this id, estimator id: {id}"
            logger.error(message)
            raise InValidRequestParameterException(({"message": message}))
        return self.get_all_registered_estimator_id()

    def get_all_registered_estimator_id(self):
        result = list()
        for item in self.estimator_by_id.keys():
            result.append(item)
        return result

    def predict_using_all_estimator(self, raw_input_data):
        prediction_by_id = dict()
        for id, estimator in self.estimator_by_id.items():
            prediction, positive_probability, negative_probability = estimator.data_preparation_and_predict(
                raw_input_data)
            prediction_by_id[id] = {
                "prediction": prediction,
                "positive_probability": positive_probability,
                "negative_probability": negative_probability
            }
        return prediction_by_id


class Estimator:
    def __init__(self, id):
        self.id = id
        self.train_task_id = None
        self.pipeline = None
        self.label_encoder_by_field_name = dict()
        self.encoding_parameters = dict()
        self.feature_engineering_parameters = dict()
        self.encoded_field_names = None
        self.time_base_field_name = None
        self.schema_name = None
        self.feature_engineered_table_name = None
        self.detailed_information_about_table = None
        self.feature_engineered_insert_script = None

    def initialize(self):
        train_task_id, pipeline, label_encoder_registry, encoding_parameters, feature_engineering_parameters, encoded_field_names, time_base_field_name, schema_name, feature_engineered_table_name, detailed_information_about_table, feature_engineered_insert_script = database_handler.get_estimator_properties(
            self.id)
        self.train_task_id = train_task_id
        self.pipeline = pipeline
        for field_name, label_encoder_id in label_encoder_registry.items():
            label_encoder = database_handler.load_label_encoder_object(label_encoder_id)
            self.label_encoder_by_field_name[field_name] = label_encoder
        self.encoding_parameters = encoding_parameters
        self.feature_engineering_parameters = feature_engineering_parameters
        self.encoded_field_names = self.remove_fraud_type_field_name(encoded_field_names)
        self.time_base_field_name = time_base_field_name
        self.schema_name = schema_name
        self.feature_engineered_table_name = feature_engineered_table_name
        self.detailed_information_about_table = detailed_information_about_table
        self.feature_engineered_insert_script = feature_engineered_insert_script

    def remove_fraud_type_field_name(self, field_names):
        for field_name, encoding_type in self.encoding_parameters.items():
            if encoding_type == FRAUD_TYPE:
                field_names.remove(field_name)
        return field_names

    def data_preparation_and_predict(self, raw_input_data):
        input_data = self.data_preprocess(raw_input_data)
        data_set = self.encode(input_data)
        if self.feature_engineering_parameters is not None:
            data_set = self.feature_engineering(data_set)
        prediction, positive_probability, negative_probability = self.predict(data_set)
        return prediction, positive_probability, negative_probability

    def data_preprocess(self, raw_input_data):
        preprocessed_inpud_data = dict()
        field_names_in_input_data = list(raw_input_data.keys())
        required_input_field_names = self.get_required_input_field_names()
        is_input_contains_all_required_field_names = self.check_if_input_contains_required_field_names(
            required_input_field_names, field_names_in_input_data)
        if not is_input_contains_all_required_field_names:
            message = "There is a missing input field name in the request"
            logger.error(message)
            raise MissingInputFieldNameException(({"message": message}))
        required_input_field_name_by_type = self.get_required_input_field_name_by_type()
        for required_input_field_name, required_type_in_python in required_input_field_name_by_type.items():
            current_field_value = raw_input_data[required_input_field_name]
            preprocessed_inpud_data[required_input_field_name] = self.convert_to_own_type(required_input_field_name,
                                                                                          current_field_value,
                                                                                          required_type_in_python)
        return preprocessed_inpud_data

    def check_if_input_contains_required_field_names(self, required_field_names, input_field_names):
        return all(i in input_field_names for i in required_field_names)

    def get_required_input_field_names(self):
        return list(self.get_required_input_field_name_by_type().keys())

    def get_required_input_field_name_by_type(self):
        field_properties = self.detailed_information_about_table.get("fields")
        chosen_fraud_type_field_name = None
        for field_name_in_encoding_parameters, planned_encoding_type in self.encoding_parameters.items():
            if planned_encoding_type == FRAUD_TYPE:
                chosen_fraud_type_field_name = field_name_in_encoding_parameters
        field_names_without_id_and_fraud_type_by_field_type = dict()
        for field_property in field_properties:
            field_name = field_property.get(NAME)
            field_type_in_database = field_property.get(TYPE)
            if field_name != ID and field_name != chosen_fraud_type_field_name:
                field_names_without_id_and_fraud_type_by_field_type[field_name] = self.convert_to_python_data_type(
                    field_type_in_database)
        return field_names_without_id_and_fraud_type_by_field_type

    def convert_to_python_data_type(self, field_type_in_database):
        database_python_data_type_mapping = {
            "int": "int",
            "float": "float",
            "varchar": "str",
            "datetime": "datetime"
        }
        data_type_in_python = database_python_data_type_mapping.get(field_type_in_database)
        return data_type_in_python

    def convert_to_own_type(self, current_field_name, current_field_value, required_type):
        if required_type == INT:
            try:
                converted_value = int(current_field_value)
            except ValueError as e:
                message = f"can not convert raw input data to integer , field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif required_type == FLOAT:
            try:
                converted_value = float(current_field_value)
            except ValueError as e:
                message = f"can not convert raw input data to float, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif required_type == STRING:
            try:
                converted_value = str(current_field_value)
            except ValueError as e:
                message = f"can not convert raw input data to string, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif required_type == DATETIME:
            try:
                converted_value = datetime.fromisoformat(current_field_value)
            except ValueError as e:
                message = f"can not convert raw input data to datetime, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        else:
            message = "unknow conversion type in preprocessing step"
            logger.error(message)
            raise NotImplementedConversionTypeException({{"message": message}})
        return converted_value

    def encode(self, input_data):
        encoded_field_values = list()
        for field_name in self.encoded_field_names:
            current_data = input_data.get(field_name)
            current_conversion_type = self.encoding_parameters.get(field_name)
            encoded_value = self.encode_single_field_value(field_name, current_data, current_conversion_type)
            encoded_field_values.append(encoded_value)
        return encoded_field_values

    def encode_single_field_value(self, current_field_name, current_data, current_conversion_type):
        if current_conversion_type is None:
            result = current_data
        elif current_conversion_type == INT:
            try:
                result = int(current_data)
            except ValueError as e:
                message = f"Can not convert to integer in encoding step, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif current_conversion_type == FLOAT:
            try:
                result = float(current_data)
            except ValueError as e:
                message = f"Can not convert to float in encoding step, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif current_conversion_type == LABEL_ENCODER:
            label_encoder = self.label_encoder_by_field_name.get(current_field_name)
            try:
                converted_value = label_encoder.transform([current_data])
                result = converted_value[0]
            except Exception as e:
                message = f"can not convert to int using label encoder in encoding step, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        elif current_conversion_type == JULIAN:
            try:
                ts = pd.Timestamp(current_data)
                result = ts.to_julian_date()
            except Exception as e:
                message = f"can not convert to Julian date in encoding step, field name: {current_field_name}"
                logger.error(message)
                raise InValidRequestParameterException({"message": message})
        else:
            message = "unknow conversion type in encoding step"
            logger.error(message)
            raise NotImplementedConversionTypeException({"message": message})
        return result

    def feature_engineering(self, encoded_input_data):
        index_of_time_base_field = self.encoded_field_names.index(self.time_base_field_name)
        time_of_input_data = encoded_input_data[index_of_time_base_field]
        index_of_card_number = self.encoded_field_names.index(CARD_NUMBER)
        card_number_value = encoded_input_data[index_of_card_number]
        feature_to_be_engineered = self.feature_engineering_parameters.get(CHOSEN_FEATURE_FIELD_FOR_FEATURE_ENGINEERING)
        index_of_feature_to_be_engineered = self.encoded_field_names.index(feature_to_be_engineered)
        intervals = self.feature_engineering_parameters.get(INTERVALS)
        previous_feature_and_time_base_array_by_intervals = dict()
        for interval in intervals:
            retrospective_time = time_of_input_data - interval
            previous_feature_and_time_base_array = database_handler.get_feature_to_be_engineered_and_time_base_field_value_for_feature_engineering(
                self.schema_name,
                self.feature_engineered_table_name,
                feature_to_be_engineered,
                self.time_base_field_name,
                # time_of_input_data,
                retrospective_time,
                card_number_value)
            previous_feature_and_time_base_array_by_intervals[interval] = previous_feature_and_time_base_array
        engineer = engineering_module.Engineer(self.time_base_field_name, self.encoded_field_names,
                                               self.feature_engineering_parameters)
        engineered_data_set = engineer.process(encoded_input_data, previous_feature_and_time_base_array_by_intervals,
                                               index_of_feature_to_be_engineered, index_of_time_base_field)
        database_handler.persist_feature_engineered_record(self.feature_engineered_insert_script, engineered_data_set)
        return engineered_data_set

    def predict(self, data_set):
        try:
            current_prediction = self.pipeline.predict([data_set])
            current_probability = self.pipeline.predict_proba([data_set])
            prediction = int(current_prediction[0])
            positive_probability = float(current_probability[0][1])
            negative_probability = float(current_probability[0][0])
            return prediction, positive_probability, negative_probability
        except Exception as e:
            print(e)


if __name__ == "__main__":
    log_file = os.getenv("ESTIMATOR_LOG_FILE", "estimator.log")
    log_level = os.getenv("ESTIMATOR_LOG_LEVEL", "DEBUG")
    # rabbitmq_url = os.getenv("RABBITMQ_URL", "localhost")
    database_url = os.getenv("MYSQL_DATABASE_URL", "localhost")
    database_user = os.getenv("MYSQL_USER", "root")
    database_password = os.getenv("MYSQL_ROOT_PASSWORD", "pwd")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger("estimator.server")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.debug("MYSQL_DATABASE_URL: " + database_url)
    logger.debug("MYSQL_USER :" + database_user)
    logger.debug("MYSQL_PASSWORD: " + database_password)
    database_handler = database_module.Handler(database_url, database_user, database_password)
    estimator_holder = EstimatorHolder()
    logger.debug("estimator modul started")
    server.run(host='0.0.0.0', port=8083, debug=True)
