import json

import sys
import time
from datetime import datetime

import mysql.connector
import numpy as np
import logging
import pickle

import timestamp as timestamp

import encoding_module
import engineering_module
import ddl_build_module

UNDER_SCORE = "_"
ENCODED = "encoded"
FE = "engineered"  # feature engineered
DOESNT_EXIST = "doesnt_exist"
IN_PROGRESS = "in_progress"
OTHER_APPLICABLE_ENCODING_TYPES="other_applicable_encoding_types"


class IllegalArgumentException(Exception):
    pass


class ForeignKeyConstraintViolationException(Exception):
    pass


class NoneException(Exception):
    pass


class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)


class Handler:
    def __init__(self, database_url, database_user, database_password, insert_batch_size):
        self.logger = logging.getLogger("train.server.database")
        self.database_url = database_url
        self.database_user = database_user
        self.database_password = database_password
        self.connection = self.get_connection(self.database_url, self.database_user, self.database_password)
        self.insert_batch_size = insert_batch_size

    def get_connection(self, database_url, database_user_name, database_password):
        try:
            connection = mysql.connector.connect(
                # pool_name="local",
                # pool_size=16,
                host=database_url,
                user=database_user_name,
                password=database_password)
            self.logger.debug("database connection created in train modul")
            connection.set_converter_class(NumpyMySQLConverter)
            return connection
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL connection error: {err.msg}")

    def create_common_fraud_schemas(self):

        commands = ["SQL CREATE SCHEMA common_fraud.txt",
                    "SQL CREATE TABLE planned_encoding_and_feature_engineering.txt",
                    "SQL CREATE TABLE raw_dataset.txt",
                    "SQL CREATE TABLE encoding.txt",
                    "SQL CREATE TABLE feature_engineering.txt",
                    "SQL CREATE TABLE encoded_table_registry.txt",
                    "SQL CREATE TABLE feature_engineered_table_registry.txt",
                    "SQL CREATE TABLE train_task.txt",
                    "SQL CREATE TABLE label_encoder.txt",
                    "SQL CREATE TABLE estimator.txt",
                    "SQL CREATE TABLE metrics.txt"]
        cursor = self.connection.cursor()
        for command in commands:
            try:
                file = open(command, "r")
                sql_create_script = file.read()
                cursor.execute(sql_create_script)
                self.connection.commit()
            except FileNotFoundError:
                self.logger.error(f"SQL script file not found {command}")
            except OSError:
                self.logger.error("OS Error")
            except mysql.connector.Error as err:
                self.logger.error(f"MySQL error message: {err.msg}, file: {file}")
        cursor.close()

    # 1.2
    def get_schema_names(self):
        cursor = self.connection.cursor()
        sql_select_query = "SHOW SCHEMAS"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            schema_names = list()
            for item in query_result:
                schema_names.append(item[0])
            self.logger.debug(f"schema names: {schema_names}")
            return schema_names
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 1.3
    def get_table_names_of_given_database(self, schema_name):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT distinct TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_name}'"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            table_names = list()
            for item in query_result:
                table_names.append(item[0])
            self.logger.debug(f"tables names of {schema_name} schema: {table_names}")
            return table_names
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 1.4
    def get_records(self, database_name, table_name):
        result = dict()
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM {database_name}.{table_name}"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            for record in query_result:
                fields = list()
                length = len(record)
                id = 0
                for index in range(length):
                    if index == 0:
                        id = record[index]
                    else:
                        fields.append(record[index])
                result[id] = fields
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()
        return result

    # 1.4
    def get_records_with_limit_and_offset(self, database_name, table_name, limit, offset):
        result = dict()
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM {database_name}.{table_name} limit {limit} offset {offset}"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            records = list()
            for item in query_result:
                fields = list()
                length = len(item)
                for index in range(length):
                    field = item[index]
                    type_of_field = type(field)
                    if type_of_field == datetime:
                        field = datetime.strftime(field, '%Y-%m-%d %H:%M:%S:%f')
                    fields.append(field)
                records.append(fields)
            return records
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()
        return result

    # 1.5
    def get_detailed_information_about_table(self, schema_name, table_name):
        result = dict()
        cursor = self.connection.cursor()
        column_properties = list()
        if table_name is None:
            self.logger.debug("No table name set, query the transaction table")
            table_name = "transaction"
        sql_select_query = f"SELECT COLUMN_NAME, DATA_TYPE  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}' ORDER BY ordinal_position"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            if query_result:
                for column_name_and_type in query_result:
                    column_property = dict()
                    column_name = column_name_and_type[0]
                    column_type_as_byte = column_name_and_type[1]
                    column_type = column_type_as_byte.decode("utf-8")
                    column_property["name"] = column_name
                    column_property["type"] = column_type
                    column_properties.append(column_property)
            result["fields"] = column_properties
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

        sql_select_query = f"SELECT COUNT(*) FROM {schema_name}.{table_name}"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchone()
            result["record_number"] = query_result[0] if query_result else 0
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

        sql_select_query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}' AND COLUMN_KEY='PRI'"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchone()
            result["primary_key"] = query_result[0] if query_result else ""
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()

        fraud_candidates = self.get_fraud_candidate_column_and_fraud_number(schema_name, table_name,
                                                                            result)
        result["fraud_candidates"] = fraud_candidates
        self.get_other_proper_column_data_type_to_encoding(schema_name, table_name, result)
        return result

    # 1.5.1
    def get_fraud_candidate_column_and_fraud_number(self, schema_name, table_name, database_property_holder):
        candidate_fraud_columns = list()
        cursor = self.connection.cursor()
        column_names_and_types = database_property_holder.get("fields")
        for column_name_and_type in column_names_and_types:
            column_name = column_name_and_type.get("name")
            sql_select_query = f"SELECT {column_name} FROM {schema_name}.{table_name}"
            try:
                cursor.execute(sql_select_query)
                query_result = cursor.fetchall()
                candidate_fraud_column = dict()
                record_number = len(query_result)
                number_of_ones = 0
                number_of_nulls = 0
                for field in query_result:
                    if field[0] == 1 or field[0] == '1':
                        number_of_ones = number_of_ones + 1
                    elif field[0] == 0 or field[0] == '0':
                        number_of_nulls = number_of_nulls + 1
                if number_of_ones + number_of_nulls == record_number and number_of_ones != 0 and number_of_nulls != 0:
                    candidate_fraud_column["name"] = column_name
                    candidate_fraud_column["fraud_number"] = number_of_ones
                    candidate_fraud_column["no_fraud_number"] = number_of_nulls
                    candidate_fraud_columns.append(candidate_fraud_column)
            except mysql.connector.Error as err:
                self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()
        return candidate_fraud_columns

    # 1.5.2
    def get_other_proper_column_data_type_to_encoding(self, schema_name, table_name, database_property_holder):
        cursor = self.connection.cursor()
        column_names_and_types = database_property_holder.get("fields")
        for column_name_and_type in column_names_and_types:
            column_name = column_name_and_type.get("name")
            column_type = column_name_and_type.get("type")
            other_applicable_encoding_types = list()
            if column_type.startswith("varchar"):
                other_applicable_encoding_types.clear()
                sql_select_query = f"SELECT {column_name} FROM {schema_name}.{table_name}"
                try:
                    cursor.execute(sql_select_query)
                    records = cursor.fetchall()
                    is_records_number_type = True
                    is_records_suited_int_type = True
                    for record_as_tuple in records:
                        record = record_as_tuple[0]
                        if not record.isnumeric():
                            is_records_number_type = False
                            is_records_suited_int_type = False
                            break
                        elif self.is_int(record):
                            value = int(record)
                            if value > 2147483647:
                                is_records_suited_int_type = False
                            else:
                                continue
                        elif self.is_float(record):
                            is_records_suited_int_type = False

                    if is_records_number_type:
                        other_applicable_encoding_types.append("float")
                    if is_records_suited_int_type:
                        other_applicable_encoding_types.append("int")
                except mysql.connector.Error as err:
                    self.logger.error(f"MySQL error message: {err.msg}")
            column_name_and_type[OTHER_APPLICABLE_ENCODING_TYPES] = other_applicable_encoding_types
        cursor.close()

    # 1.5.3
    def is_int(self, element):
        try:
            int(element)
            return True
        except ValueError:
            return False

    # 1.5.4
    def is_float(self, element):
        try:
            float(element)
            return True
        except ValueError:
            return False

    # 2.1
    def persist_encoding_and_engineering_plan(self, schema_name, table_name, detailed_information_about_table,
                                              used_time_base_field_name,
                                              encoding_parameters,
                                              feature_engineering_parameters):
        cursor = self.connection.cursor()
        sql_insert_query = "INSERT INTO common_fraud.planned_encoding_and_feature_engineering (schema_name, table_name, detailed_information_about_table, time_base_field_name ,encoding_parameters, feature_engineering_parameters) VALUES (%s,%s,%s,%s,%s,%s)"
        parameter = (
            schema_name, table_name, json.dumps(detailed_information_about_table), used_time_base_field_name,
            json.dumps(encoding_parameters),
            json.dumps(feature_engineering_parameters))
        try:
            cursor.execute(sql_insert_query, parameter)
            last_inserted_row_id = cursor.lastrowid
            self.connection.commit()
            self.logger.debug(
                f"encoding and feature enginnering plan persisted, schema name: {schema_name}, table name: {table_name}, encoding parameters: {encoding_parameters}, feature engineering parameters {feature_engineering_parameters}")
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()
        return last_inserted_row_id

    # 2.1
    def get_all_encoding_and_feature_engineering_plan(self):
        response = dict()
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.planned_encoding_and_feature_engineering"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        response = None
        if query_result is not None:
            response = self.build_encoding_and_feature_engineering_plan_response(query_result)
        return response

    # 2.1
    def get_encoding_and_feature_engineering_plan_by_id(self, id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.planned_encoding_and_feature_engineering WHERE id = %s"
        parameter = (id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        if query_result is None:
            raise NoneException({"message": "The given id doesn't exist", "parameter": id})
        response = self.build_encoding_and_feature_engineering_plan_response(query_result)
        return response

    def delete_encoding_and_feature_engineering_plan(self, id):
        cursor = self.connection.cursor()
        sql_delete_command = "DELETE FROM common_fraud.planned_encoding_and_feature_engineering WHERE id= %s"
        parameter = (id,)
        try:
            cursor.execute(sql_delete_command, parameter)
            self.connection.commit()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 2.1
    def build_encoding_and_feature_engineering_plan_response(self, query_result):
        response = dict()
        for record in query_result:
            single_response = dict()
            id = record[0]
            single_response["schema_name"] = record[1]
            single_response["table_name"] = record[2]
            single_response["detailed_information_about_table"] = json.loads(record[3])
            single_response["time_base_field_name"] = record[4]
            single_response["encoding_parameters"] = json.loads(record[5])
            single_response["feature_engineering_parameters"] = json.loads(record[6])
            response[id] = single_response
        return response

    # 2.2
    def persist_train_task(self, request_parameter, feature_selector_keys, samplers_keys, scalers_keys, models_keys):
        cursor = self.connection.cursor()
        planned_encoding_and_feature_engineering_id = request_parameter.get(
            "planned_encoding_and_feature_engineering_id")
        if planned_encoding_and_feature_engineering_id is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body",
                 "parameter_name": "planned_encoding_and_feature_engineering_id"})
        existing_planned_encoding_id_and_feature_engineering_id = self.get_existing_planned_encoding_and_feature_engineering_id()
        if planned_encoding_and_feature_engineering_id not in existing_planned_encoding_id_and_feature_engineering_id:
            raise IllegalArgumentException(
                {"message": "Parameter is invalid", "parameter_name": "planned_encoding_and_feature_engineering_id",
                 "value": planned_encoding_and_feature_engineering_id})

        feature_selector_code = request_parameter.get("feature_selector_code")
        if feature_selector_code is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body", "parameter_name": "feature selector code"})
        if feature_selector_code not in feature_selector_keys:
            raise IllegalArgumentException(
                {"message": "Parameter is invalid", "parameter_name": "feature_selector_code",
                 "value": feature_selector_code})

        sampler_code = request_parameter.get("sampler_code")
        if sampler_code is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body", "parameter_name": "sampler_code"})
        if sampler_code not in samplers_keys:
            raise IllegalArgumentException(
                {"message": "Parameter is invalid", "parameter_name": "sampler_code",
                 "value": sampler_code})

        scaler_code = request_parameter.get("scaler_code")
        if scaler_code is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body", "parameter_name": "scaler_code"})
        if scaler_code not in scalers_keys:
            raise IllegalArgumentException(
                {"message": "Parameter is invalid", "parameter_name": "scaler_code",
                 "value": scaler_code})

        model_code = request_parameter.get("model_code")
        if model_code is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body", "parameter_name": "model_code"})
        if model_code not in models_keys:
            raise IllegalArgumentException(
                {"message": "Parameter is invalid", "parameter_name": "model_code",
                 "value": model_code})

        test_size = request_parameter.get("test_size")
        if test_size is None:
            raise IllegalArgumentException(
                {"message": "Parameter is missing in request body", "parameter_name": "test_size"})
        if test_size >= 1 or test_size <= 0:
            raise IllegalArgumentException(
                {"message": "Parameter must be between 0 and 1", "parameter_name": "test_size", "value": test_size})

        progress_status = "SAVED"

        self.check_if_train_task_exist(planned_encoding_and_feature_engineering_id, feature_selector_code,
                                       sampler_code, scaler_code,
                                       model_code, test_size)

        sql_insert_query = "INSERT INTO common_fraud.train_task (planned_encoding_and_feature_engineering_id, feature_selector_code, sampler_code, scaler_code, model_code, test_size, progress_status) VALUES (%s,%s,%s,%s,%s,%s,%s)"
        parameter = (
            planned_encoding_and_feature_engineering_id, feature_selector_code, sampler_code, scaler_code,
            model_code, test_size, progress_status)
        try:
            cursor.execute(sql_insert_query, parameter)
            self.connection.commit()
            last_inserted_row_id = cursor.lastrowid
            self.logger.debug(
                f"Train task persisted, planned_encoding_id_and_feature_engineering_id: {planned_encoding_and_feature_engineering_id}, feature_selector_code: {feature_selector_code}, sampler_code: {sampler_code}, scaler_code: {scaler_code}, model_code: {model_code}, test_size: {test_size}")
            cursor.close()
            return last_inserted_row_id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def check_if_train_task_exist(self, planned_encoding_and_feature_engineering_id, feature_selector_code,
                                  sampler_code, scaler_code, model_code, test_size):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.train_task WHERE planned_encoding_and_feature_engineering_id = %s and feature_selector_code = %s and sampler_code = %s and scaler_code = %s and model_code = %s and test_size = %s"
        parameter = (
            planned_encoding_and_feature_engineering_id, feature_selector_code, sampler_code, scaler_code, model_code,
            test_size)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            if query_result is not None:
                id = query_result[0]
                raise IllegalArgumentException({"message": "Train task exists yet", "id": id})
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 2.2
    def get_all_train_task(self):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.train_task"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            result = self.build_train_task_response(query_result)
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 2.2
    def get_train_task_by_id(self, id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.train_task WHERE id = %s"
        parameter = (id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchall()
            cursor.close()
            result = self.build_train_task_response(query_result)
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 2.2
    def build_train_task_response(self, query_result):
        result = dict()
        for item in query_result:
            id = item[0]
            planned_encoding_and_feature_engineering_id = item[1]
            feature_selector_code = item[2]
            sampler_code = item[3]
            scaler_code = item[4]
            model_code = item[5]
            test_size = item[6]
            process_status = item[7]
            result[id] = {
                "planned_encoding_and_feature_engineering_id": planned_encoding_and_feature_engineering_id,
                "feature_selector_code": feature_selector_code,
                "sampler_code": sampler_code,
                "scaler_code": scaler_code,
                "model_code": model_code,
                "test_size": test_size,
                "process_status": process_status}
        return result

    def set_train_task_status(self, id, progress_status):
        cursor = self.connection.cursor()
        sql_update_command = "UPDATE common_fraud.train_task SET progress_status = %s WHERE id = %s"
        parameter = (progress_status, id)
        try:
            cursor.execute(sql_update_command, parameter)
            self.connection.commit()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def get_train_task_status(self, id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT progress_status FROM common_fraud.train_task WHERE id = %s"
        parameter = (id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            cursor.close()
            if query_result is not None:
                status = query_result[0]
            else:
                status = DOESNT_EXIST
            return status
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def check_if_any_train_in_progress(self):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.train_task WHERE progress_status = %s"
        parameter = (IN_PROGRESS,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchall()
            cursor.close()
            if query_result:
                result = True
            else:
                result = False
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 2.2.1
    def get_existing_planned_encoding_and_feature_engineering_id(self):
        cursor = self.connection.cursor()
        result = list()
        sql_select_query = "select id from common_fraud.planned_encoding_and_feature_engineering"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        for item in query_result:
            result.append(item[0])
        return result

    # 3.1.1
    def get_parameters_from_plan_by_id(self, id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.planned_encoding_and_feature_engineering WHERE id = %s"
        parameter = (id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        schema_name = query_result[1]
        table_name = query_result[2]
        detailed_information_about_table = json.loads(query_result[3])
        time_base_field_name = query_result[4]
        encoding_parameters = json.loads(query_result[5])
        feature_engineering_parameters = json.loads(query_result[6])
        return schema_name, table_name, detailed_information_about_table, time_base_field_name, encoding_parameters, feature_engineering_parameters

    # 3.1.2
    def get_raw_dataset_id(self, schema_name, table_name):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.raw_dataset WHERE schema_name = %s and table_name = %s"
        parameter = (schema_name, table_name)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            if query_result is not None:
                id = query_result[0]
            else:
                sql_insert_command = "INSERT INTO  common_fraud.raw_dataset (schema_name, table_name) VALUES(%s, %s)"
                parameter = (schema_name, table_name)
                cursor.execute(sql_insert_command, parameter)
                self.connection.commit()
                id = cursor.lastrowid
                cursor.close()
            return id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.3
    def get_encoding_id_by_json_content(self, encoding_parameters):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.encoding"
        try:
            id = None
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            for item in query_result:
                if json.loads(item[1]) == encoding_parameters:
                    id = item[0]
            if id is None:
                sql_insert_command = "INSERT INTO  common_fraud.encoding (encoding_parameters) VALUES(%s)"
                parameter = (json.dumps(encoding_parameters),)
                cursor.execute(sql_insert_command, parameter)
                self.connection.commit()
                id = cursor.lastrowid
                cursor.close()
            return id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.4.1
    def get_encoded_table_properties(self, raw_dataset_id, encoding_id, time_base_field_name):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.encoded_table_registry where raw_dataset_id = %s and encoding_id = %s and time_base_field_name = %s"
        try:
            parameter = (raw_dataset_id, encoding_id, time_base_field_name)
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            id = None
            encoded_table_name = None
            encoded_field_names = None
            label_encoder_registry = None
            if query_result is not None:
                id = query_result[0]
                encoded_table_name = query_result[4]
                encoded_field_names = json.loads(query_result[5])
                label_encoder_registry = json.loads(query_result[6])
            return id, encoded_table_name, encoded_field_names, label_encoder_registry
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.6.2
    def get_feature_engineering_id_by_json_content(self, feature_engineering_parameters):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.feature_engineering"
        try:
            id = None
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            for item in query_result:
                if json.loads(item[1]) == feature_engineering_parameters:
                    id = item[0]
            if id is None:
                sql_insert_command = "INSERT INTO  common_fraud.feature_engineering (feature_engineering_parameters) VALUES(%s)"
                parameter = (json.dumps(feature_engineering_parameters),)
                cursor.execute(sql_insert_command, parameter)
                self.connection.commit()
                id = cursor.lastrowid
                cursor.close()
            return id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.5
    def encode(self, raw_dataset_id, schema_name, table_name, encoding_id, encoding_parameters, time_base_field_name):

        detailed_information_about_table = self.get_detailed_information_about_table(schema_name, table_name)
        field_types_by_name = self.get_field_types_by_name(schema_name, table_name)
        original_field_names=list(field_types_by_name.keys())
        original_data_set = self.get_record_for_processing(schema_name, table_name, time_base_field_name)
        encoder = encoding_module.DataBaseEncoder(original_data_set, original_field_names)
        encoded_array, encoder_by_field_name, encoded_field_names = encoder.encode(encoding_parameters,
                                                                                   detailed_information_about_table)
        encoded_table_name = table_name + UNDER_SCORE + ENCODED + UNDER_SCORE + str(encoding_id)
        ddl_command_builder = ddl_build_module.DdlCommandBuilder(schema_name)
        encoded_table_create_script = ddl_command_builder.build_create_encoded_or_engineered_table_script(
            encoded_field_names,
            encoded_table_name)
        self.create_encoded_or_engineered_table(encoded_table_create_script)
        insert_script = ddl_command_builder.create_insert_into_encoded_or_feature_engineered_script(encoded_field_names,
                                                                                                    encoded_table_name)
        self.persist_encoded_or_engineered_dataset(insert_script, encoded_array, type="Encoded")
        label_encoder_registry = dict()
        for field_name, label_encoder in encoder_by_field_name.items():
            id_in_label_encoder_table = self.persist_label_encoder(field_name, label_encoder)
            label_encoder_registry[field_name] = id_in_label_encoder_table
        encoded_table_registry_id = self.persist_encoded_table_registry(raw_dataset_id, encoding_id,
                                                                        time_base_field_name, encoded_table_name,
                                                                        encoded_field_names, label_encoder_registry)
        return encoded_array, encoded_table_registry_id, encoded_table_name, encoded_field_names, label_encoder_registry

    # 3.1.5
    def get_field_types_by_name(self, schema_name, table_name):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT COLUMN_NAME, DATA_TYPE  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}' ORDER BY ordinal_position"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            field_types_by_name=dict()
            for column_name_and_type in query_result:
                field_name = column_name_and_type[0]
                field_type=column_name_and_type[1].decode("utf-8")
                field_types_by_name[field_name]=field_type
            cursor.close()
            self.logger.debug(f"field types by name in {schema_name} schema, {table_name} table: {field_types_by_name}")
            return field_types_by_name
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def get_record_for_processing(self, schema_name, table_name, used_time_base_field):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM {schema_name}.{table_name} ORDER BY {used_time_base_field} DESC"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            result = np.array(query_result)
            cursor.close()
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.5.7
    def persist_encoded_or_engineered_dataset(self, script, dataset, type):
        cursor = self.connection.cursor()
        length_of_dataset = len(dataset)
        bound = self.insert_batch_size
        try:
            if length_of_dataset > bound:
                number_of_part_array = int(length_of_dataset / bound)
                number_of_rest_datas = length_of_dataset - number_of_part_array * bound
                for i in range(0, number_of_part_array, 1):
                    temp_array = dataset[i * bound:(i + 1) * bound, :]
                    value_list = list()
                    for record in temp_array:
                        value_list.append(tuple(record))
                    cursor.executemany(script, value_list)
                    self.connection.commit()
                temp_array = dataset[
                             (number_of_part_array) * bound:(number_of_part_array) * bound + number_of_rest_datas, :]
                value_list = list()
                for record in temp_array:
                    value_list.append(tuple(record))
                cursor.executemany(script, value_list)
                self.connection.commit()
            else:
                value_list = list()
                for record in dataset:
                    value_list.append(tuple(record))
                cursor.executemany(script, value_list)
                self.connection.commit()
            cursor.close()
            self.logger.debug(f"{type} database persisted")
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.6
    def get_table_name_from_feature_engineering_registry(self, encoded_table_registry_id, feature_engineering_id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.feature_engineered_table_registry WHERE encoded_table_registry_id = %s and feature_engineering_id = %s"
        try:
            parameter = (encoded_table_registry_id, feature_engineering_id)
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            feature_engineered_table_name = None
            feature_engineered_insert_script = None
            if query_result is not None:
                feature_engineered_table_name = query_result[3]
                feature_engineered_insert_script = query_result[4]
            return feature_engineered_table_name, feature_engineered_insert_script
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.5.3
    def create_encoded_or_engineered_table(self, script):
        cursor = self.connection.cursor()
        try:
            cursor.execute(script)
            self.connection.commit()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 3.1.5.4
    def get_records_for_processing(self, schema_name, table_name, time_stamp_field_name):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM {schema_name}.{table_name} ORDER BY {time_stamp_field_name} DESC"
        try:
            cursor.execute(sql_select_query)
            result = cursor.fetchall()
            cursor.close()
            records_with_id = np.array(result)
            records_without_id = records_with_id[:, 1:]
            return records_without_id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def persist_label_encoder(self, field_name, encoder):
        cursor = self.connection.cursor()
        sql_insert_command = "INSERT INTO common_fraud.label_encoder (encoder_object) values (%s)"
        pickled_encoder = pickle.dumps(encoder)
        parameter = (pickled_encoder,)
        try:
            cursor.execute(sql_insert_command, parameter)
            last_inserted_row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            self.logger.debug(f"label encoder persisted, field name: {field_name}, id: {id}")
            return last_inserted_row_id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def persist_encoded_table_registry(self, raw_dataset_id, encoding_id, time_base_field_name, encoded_table_name,
                                       encoded_field_names,
                                       label_encoder_registry):
        cursor = self.connection.cursor()
        sql_insert_command = "INSERT INTO common_fraud.encoded_table_registry (raw_dataset_id, encoding_id, time_base_field_name, encoded_table_name, encoded_field_names, label_encoder_registry) values (%s,%s,%s,%s,%s,%s)"
        parameter = (
            raw_dataset_id, encoding_id, time_base_field_name, encoded_table_name, json.dumps(encoded_field_names),
            json.dumps(label_encoder_registry))
        try:
            cursor.execute(sql_insert_command, parameter)
            last_inserted_row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            self.logger.debug(
                f"record persisted in the encoded_table_registry table, encoded_table_name: {encoded_table_name}")
            return last_inserted_row_id
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def feature_engineer(self, schema_name, encoded_table_name, fraud_type_field_name, feature_engineering_id, dataset,
                         time_base_field_name,
                         encoded_field_names, feature_engineering_parameters, encoded_table_registry_id, used_cpu_core):
        engineer = engineering_module.Engineer(time_base_field_name, encoded_field_names, fraud_type_field_name,
                                               feature_engineering_parameters,
                                               used_cpu_core)
        feature_engineered_dataset, feature_engineered_field_names = engineer.create_new_features(dataset)
        feature_engineered_table_name = encoded_table_name + UNDER_SCORE + FE + UNDER_SCORE + str(
            feature_engineering_id)
        ddl_command_builder = ddl_build_module.DdlCommandBuilder(schema_name)
        feature_engineered_table_create_script = ddl_command_builder.build_create_encoded_or_engineered_table_script(
            feature_engineered_field_names,
            feature_engineered_table_name)
        self.create_encoded_or_engineered_table(feature_engineered_table_create_script)
        feature_engineered_insert_script = ddl_command_builder.create_insert_into_encoded_or_feature_engineered_script(
            feature_engineered_field_names, feature_engineered_table_name)
        feature_engineered_insert_script_using_in_prediction = ddl_command_builder.create_insert_into_encoded_or_feature_engineered_script_for_prediction(
            feature_engineered_field_names, feature_engineered_table_name, fraud_type_field_name)
        self.persist_encoded_or_engineered_dataset(feature_engineered_insert_script, feature_engineered_dataset,
                                                   type="Feature engineered")
        self.persist_feature_engineered_table_registry_item(encoded_table_registry_id, feature_engineering_id,
                                                            feature_engineered_table_name,
                                                            feature_engineered_insert_script)
        return feature_engineered_dataset, feature_engineered_table_name, feature_engineered_insert_script_using_in_prediction

    def persist_feature_engineered_table_registry_item(self, encoded_table_registry_id, feature_engineering_id,
                                                       feature_engineered_table_name, feature_engineered_insert_script):
        cursor = self.connection.cursor()
        sql_insert_command = "INSERT INTO common_fraud.feature_engineered_table_registry (encoded_table_registry_id, feature_engineering_id, feature_engineered_table_name, feature_engineered_insert_script) values (%s,%s,%s,%s)"
        parameter = (encoded_table_registry_id, feature_engineering_id, feature_engineered_table_name,
                     feature_engineered_insert_script)
        try:
            cursor.execute(sql_insert_command, parameter)
            self.connection.commit()
            cursor.close()
            self.logger.debug(
                f"record persisted in feature_engineered_table_registry table, feature_engineered_table_name: {feature_engineered_table_name}")
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    #
    def get_train_parameters(self, train_task_id):
        cursor = self.connection.cursor()
        sql_select_query = "select * from common_fraud.train_task where id = %s"
        parameter = (train_task_id,)
        try:
            cursor.execute(sql_select_query, parameter)
            result = cursor.fetchone()
            cursor.close()
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    # 5
    def persist_estimator(self, train_task_id, estimator):
        cursor = self.connection.cursor()
        sql_insert_query = "INSERT INTO common_fraud.estimator (train_task_id, estimator_object) VALUES (%s,%s)"
        pickled_estimator_container = pickle.dumps(estimator)
        parameter = (train_task_id, pickled_estimator_container)
        try:
            cursor.execute(sql_insert_query, parameter)
            last_inserted_row_id = cursor.lastrowid
            self.connection.commit()
            self.logger.debug(f"estimator persisted, train_task_id: {train_task_id}, id: {last_inserted_row_id}")
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()
        return last_inserted_row_id

    # 4
    def persist_metrics(self, estimator_id, TP, FP, TN, FN, sensitivity, specificity, accuracy, balanced_accuracy,
                        precision, recall, PPV, NPV, FNR, FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC, Youdens_statistic):
        cursor = self.connection.cursor()
        sql_insert_query = "INSERt INTO common_fraud.metrics (estimator_id,TP,FP,TN,FN,sensitivity,specificity,accuracy,balanced_accuracy,prec,recall,PPV,NPV,FNR,FPR,FDR,F_OR,f1,f_05,f2,MCC,ROCAUC,Youdens_statistic) VALUES" \
                           "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        values = (estimator_id, TP, FP, TN, FN, sensitivity, specificity, accuracy, balanced_accuracy,
                  precision, recall, PPV, NPV, FNR, FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC, Youdens_statistic)
        try:
            cursor.execute(sql_insert_query, values)
            last_inserted_row_id = cursor.lastrowid
            self.connection.commit()
            self.logger.debug(f"metrics persisted, estimator_id: {estimator_id}, id: {last_inserted_row_id}")
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        cursor.close()

    def get_all_metrics(self):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.metrics"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        single_response = self.build_metrics_response(query_result)
        return single_response

    # 6
    def get_metrics_by_estimator_id(self, estimator_id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.metrics WHERE estimator_id = %s"
        parameter = (estimator_id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        single_response = self.build_metrics_response(query_result)
        return single_response

    def build_metrics_response(self, query_result):
        response = list()
        for item in query_result:
            single_result = dict()
            single_result["id"] = item[0]
            single_result["estimator_id"] = item[1]
            single_result["TP"] = item[2]
            single_result["FP"] = item[3]
            single_result["TN"] = item[4]
            single_result["FN"] = item[5]
            single_result["sensitivity"] = item[6]
            single_result["specificity"] = item[7]
            single_result["accuracy"] = item[8]
            single_result["balanced_accuracy"] = item[9]
            single_result["precision"] = item[10]
            single_result["recall"] = item[11]
            single_result["PPV"] = item[12]
            single_result["NPV"] = item[13]
            single_result["FNR"] = item[14]
            single_result["FPR"] = item[15]
            single_result["FDR"] = item[16]
            single_result["FOR"] = item[17]
            single_result["f1"] = item[18]
            single_result["f0.5"] = item[19]
            single_result["f2"] = item[20]
            single_result["MCC"] = item[21]
            single_result["ROCAUC"] = item[22]
            single_result["Youdens_statistic"] = item[23]
            response.append(single_result)
        return response

    def get_all_estimator(self):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.estimator"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            response = dict()
            if query_result:
                response = self.build_estimator_response(query_result)
            return response
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
        single_response = self.build_metrics_response(query_result)
        return single_response

    def get_estimator_by_train_task_id(self, train_task_id):
        cursor = self.connection.cursor()
        sql_select_query = "SELECT * FROM common_fraud.estimator WHERE train_task_id = %s"
        parameter = (train_task_id,)
        try:
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchall()
            cursor.close()
            response = dict()
            if query_result:
                response = self.build_estimator_response(query_result)
            return response
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def build_estimator_response(self, query_result):
        summarized_result = list()
        for item in query_result:
            single_result=dict()
            single_result["id"] =  item[0]
            single_result["train_task_id"] = item[1]
            summarized_result.append(single_result)
        return summarized_result


