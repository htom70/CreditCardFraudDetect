import json
import logging
import mysql.connector
import numpy as np
import pickle
import sklearn
import lightgbm


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
    def __init__(self, database_url, database_user, database_password):
        self.logger = logging.getLogger("estimator.server.database")
        self.database_url = database_url
        self.database_user = database_user
        self.database_password = database_password
        self.connection = self.get_connection(self.database_url, self.database_user, self.database_password)

    def get_connection(self, database_url, database_user_name, database_password):
        try:
            connection = mysql.connector.connect(
                # pool_name="local",
                # pool_size=16,
                host=database_url,
                user=database_user_name,
                password=database_password)
            self.logger.debug("database connection created in estimator modul")
            connection.set_converter_class(NumpyMySQLConverter)
            return connection
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL connection error: {err.msg}")

    def get_schemas(self):
        cursor = self.connection.cursor()
        sql_select_query = f"SHOW SCHEMAS"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            result = self.build_result_from_query_result(query_result)
            return result if query_result is not None else None
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def get_tables_of_common_fraud(self, schema_name):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT distinct TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{schema_name}'"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            result = self.build_result_from_query_result(query_result)
            return result if query_result is not None else None
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def build_result_from_query_result(self, query_result):
        result = list()
        for item in query_result:
            temp = item[0]
            result.append(temp)
        return result

    def get_available_estimator_ids(self):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT id FROM common_fraud.estimator"
        try:
            cursor.execute(sql_select_query)
            query_result = cursor.fetchall()
            cursor.close()
            response = list()
            if query_result is not None:
                for item in query_result:
                    response.append(item[0])
            return response if query_result is not None else None
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def get_estimator_properties(self, id):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM common_fraud.estimator WHERE id = %s"
        try:
            parameter = (id,)
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            train_task_id = query_result[1]
            estimator_properties = pickle.loads(query_result[2])
            pipeline = estimator_properties.get("pipeline")
            label_encoder_registry = estimator_properties.get("label_encoder_registry")
            encoding_parameters = estimator_properties.get("encoding_parameters")
            feature_engineering_parameters = estimator_properties.get("feature_engineering_parameters")
            encoded_field_names = estimator_properties.get("encoded_field_names")
            time_base_field_name = estimator_properties.get("time_base_field_name")
            schema_name = estimator_properties.get("schema_name")
            feature_engineered_table_name = estimator_properties.get("feature_engineered_table_name")
            detailed_information_about_table = estimator_properties.get("detailed_information_about_table")
            feature_engineered_insert_script = estimator_properties.get("feature_engineered_insert_script")
            cursor.close()
            self.logger.debug(f"estimator object loaded from database, estimator_id: {id}")
            return train_task_id, pipeline, label_encoder_registry, encoding_parameters, feature_engineering_parameters, encoded_field_names, time_base_field_name, schema_name, feature_engineered_table_name, detailed_information_about_table, feature_engineered_insert_script if query_result is not None else None
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def load_label_encoder_object(self, id):
        cursor = self.connection.cursor()
        sql_select_query = f"SELECT * FROM common_fraud.label_encoder WHERE id = %s"
        try:
            parameter = (id,)
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchone()
            cursor.close()
            result = None
            if query_result is not None:
                result = pickle.loads(query_result[1])
            return result
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def get_feature_to_be_engineered_and_time_base_field_value_for_feature_engineering(self, schema_name,
                                                                                       feature_engineered_table_name,
                                                                                       feature_to_be_engineered,
                                                                                       time_base_field_name,
                                                                                       # time_of_input_data,
                                                                                       retrospective_time,
                                                                                       card_number_value):
        cursor = self.connection.cursor()
        float_retrespective_time = float(retrospective_time)
        sql_select_query = f"SELECT {feature_to_be_engineered}, {time_base_field_name} FROM {schema_name}.{feature_engineered_table_name} WHERE card_number = %s and  {time_base_field_name} > %s ORDER BY {time_base_field_name} DESC"
        try:
            parameter = (card_number_value, float_retrespective_time)
            cursor.execute(sql_select_query, parameter)
            query_result = cursor.fetchall()
            cursor.close()
            return np.array(query_result) if query_result is not None else None
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")

    def persist_feature_engineered_record(self, insert_script, data):
        cursor = self.connection.cursor()
        try:
            cursor.execute(insert_script, data)
            self.connection.commit()
            self.logger.debug("record persisted in feature engineered table")
            cursor.close()
        except mysql.connector.Error as err:
            self.logger.error(f"MySQL error message: {err.msg}")
