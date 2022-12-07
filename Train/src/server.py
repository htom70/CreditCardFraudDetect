import json
import logging
import math
import os
import time
from threading import Thread

import pika
from flask import Flask, jsonify, request, Response
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer, RobustScaler, MaxAbsScaler, \
    MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import column_or_1d
from xgboost import XGBClassifier

import database_module

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=41691, stdoutToServer=True, stderrToServer=True)

DEFAULT_INSERT_BATCH_SIZE = 1000
DATABASE_ENCODED_YET = "database_encoded_yet"
DATABASE_ENCODING_BEGAN = "database_encoding_began"
DATABASE_ENCODING_FINISHED = "database_encoding_finished"
DATABASE_FEATURE_ENGINEERED_YET = "database_feature_engineered_yet"
DATABASE_FEATURE_ENGINEERED_BEGAN = "database_feature_engineered_began"
DATABASE_FEATURE_ENGINEERED_FINISHED = "database_feature_engineered_finished"
SAMPLING_BEGAN = "sampling_began"
SAMPLING_FINISHED = "sampling_finished"
PIPELINE_FIT_BEGAN = "pipeline_fit_began"
PIPELINE_FIT_FINISHED = "pipeline_fit_finished"
PIPELINE_PERSISTED = "pipeline_persisted"
IN_POGRESS = "in_progress"
FINISHED = "finished"
FRAUD_TYPE = "fraud_type"

is_fit_in_progress = None

app = Flask(__name__)


class InValidRequestParameterException(Exception):
    pass


class NoTimeBaseFieldInTableException(Exception):
    pass


class FieldNotApplicableForTimeBaseException(Exception):
    pass


class NoFraudTypeFieldException(Exception):
    pass


class TooManyFraudTypeFieldException(Exception):
    pass


class NoEnoughParameterException(Exception):
    pass


class FieldNotExistInTableException(Exception):
    pass


class EncodingNotApplicableException(Exception):
    pass


class FeatureEngineeringNotApplicableException(Exception):
    pass


class FieldNotApplicableForFraudException(Exception):
    pass


def send_async_message(id, message):
    response = {
        "train_task_id": id,
        "message": message
    }
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitmq_url, port=5672))
        channel = connection.channel()
        channel.queue_declare(queue='train', durable=True)
        channel.basic_publish(exchange='', routing_key='train', body=json.dumps(response))
        connection.close()
    except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as error:
        logger.error(f"RabbitMQ error: {error}")


def get_feature_selectors(cpu_core_num=1):
    feature_selectors = {
        1: ('RFE', RFECV(estimator=RandomForestClassifier(n_jobs=cpu_core_num), n_jobs=cpu_core_num)),
        2: ('PCA', PCA(n_components=0.95, svd_solver='full')),
        3: ('SVD', TruncatedSVD())
    }
    return feature_selectors


def get_samplers(cpu_core_num=1):
    samplers = {
        1: ('RandomUnderSampler', RandomUnderSampler(sampling_strategy=0.5)),
        2: ('RandomOverSampler', RandomOverSampler(sampling_strategy=0.5)),
        3: ('SMOTEENN', SMOTEENN(sampling_strategy=0.5, n_jobs=cpu_core_num))
    }
    return samplers


def get_scalers():
    scalers = {
        1: ('StandardScaler', StandardScaler()),
        2: ('MinMaxScaler', MinMaxScaler()),
        3: ('MaxAbsScaler', MaxAbsScaler()),
        4: ('RobustScaler', RobustScaler()),
        5: ('QuantileTransformer-Normal', QuantileTransformer(output_distribution='normal')),
        6: ('QuantileTransformer-Uniform', QuantileTransformer(output_distribution='uniform')),
        7: ('Normalizer', Normalizer()),
    }
    return scalers


def get_models(cpu_core_num=1):
    models = {
        1: ('Logistic Regression', LogisticRegression(n_jobs=cpu_core_num)),
        2: ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        3: ('K-Nearest Neighbor', KNeighborsClassifier(n_jobs=cpu_core_num)),
        4: ('DecisionTree', DecisionTreeClassifier()),
        5: ('GaussianNB', GaussianNB()),
        # 'SupportVectorMachine GPU': SupportVectorMachine(use_gpu=True),
        # 'Random Forest GPU': RandomForestClassifier(use_gpu=True, gpu_ids=[0, 1], use_histograms=True),
        6: ('Random Forest', RandomForestClassifier(n_jobs=cpu_core_num)),
        # 'MLP': MLPClassifier(),
        7: ('Light GBM', LGBMClassifier(n_jobs=cpu_core_num)),
        # 'XGBoost': XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        8: ('XGBoost', XGBClassifier())
    }
    return models


@app.route('/cpu', methods=['GET'])
def get_available_cpu_core():
    core_number = os.cpu_count()
    logger.debug(f"available cpu core number: {core_number}")
    return jsonify(core_number)


@app.route('/schema', methods=['GET'])
def get_schemas():
    schemas = database_handler.get_schema_names()
    return jsonify(schemas)


@app.route('/table', methods=['GET'])
def get_tables():
    schema_name = request.args.get("schema_name")
    if schema_name is None:
        message = "schema name is missing"
        logger.error(message)
        return Response(message, 400)
    tables = database_handler.get_table_names_of_given_database(schema_name)
    return jsonify(tables)


@app.route('/record', methods=['GET'])
def get_records_with_limit_and_offset():
    result = list()
    schema_name = request.args.get("schema_name")
    if schema_name is None:
        message = "Schema name is missing"
        logger.error(message)
        return Response(message, 400)
    table_name = request.args.get("table_name")
    if table_name is None:
        message = "table name is missing"
        logger.error(message)
        return Response(message, 400)
    limit = request.args.get("limit")
    if limit is not None and not is_int(limit):
        message = "limit isn't integer"
        logger.error(message)
        return Response(message, 400)
    offset = request.args.get("offset")
    if offset is not None and not is_int(offset):
        message = "offset isn't integer"
        logger.error(message)
        return Response(message, 400)
    field_types_by_name = database_handler.get_field_types_by_name(schema_name, table_name)
    logger.debug(field_types_by_name)
    result.append(field_types_by_name)
    if limit is None or offset is None:
        result.append(database_handler.get_records(schema_name, table_name))
    else:
        result.append(database_handler.get_records_with_limit_and_offset(schema_name, table_name, limit, offset))
    return jsonify(result)


@app.route('/table_info', methods=['GET'])
def get_table_info():
    schema_name = request.args.get("schema_name")
    table_name = request.args.get("table_name")
    try:
        check_schema(schema_name)
        check_table(schema_name, table_name)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message") + (
            f", schema name: {schema_name}" if schema_name is not None and schema_name != "" else "") + (
                      f", table name: {table_name}" if table_name is not None and table_name != "" else "")
        return Response(message, 400)
    detailed_information_about_table = database_handler.get_detailed_information_about_table(schema_name=schema_name,
                                                                                             table_name=table_name)
    return jsonify(detailed_information_about_table)


def check_schema(schema_name):
    if schema_name is None or schema_name == "":
        message = "schema name is missing"
        logger.error(message)
        raise InValidRequestParameterException(({"message": message}))
    schemas = database_handler.get_schema_names()
    if schema_name not in schemas:
        message = "schema doesn't exist"
        logger.error(message)
        raise InValidRequestParameterException({"message": message})


def check_table(schema_name, table_name):
    tables = database_handler.get_table_names_of_given_database(schema_name)
    if table_name is None or table_name == "":
        message = "table name is missing"
        logger.error(message)
        raise InValidRequestParameterException({"message": message})
    if table_name not in tables:
        message = "table doesn't exist"
        logger.error(message)
        raise InValidRequestParameterException({"message": message})


@app.route('/encoding_and_feature_engineering_plan', methods=['POST'])
def save_encoding_plan():
    if not request.is_json:
        error_message = "request body must be JSON"
        logger.error(error_message)
        return Response(error_message, status=415)
    request_parameter = request.get_json()
    if request_parameter is None or not request_parameter:
        error_message = "encoding plan request's POST body is empty"
        logger.error(error_message)
        return Response(error_message, status=400)
    schema_name = request_parameter.get("schema_name")
    if schema_name is None:
        error_message = "encoding plan request's schema name is missing"
        logger.error(error_message)
        return Response(error_message, status=400)
    schemas = database_handler.get_schema_names()
    if schema_name not in schemas:
        error_message = "encoding plan request's schema name doesn't exist in database"
        logger.error(error_message)
        return Response(error_message, status=400)
    table_name = request_parameter.get("table_name")
    if table_name is None:
        error_message = "encoding plan request's table name is missing"
        logger.error(error_message)
        return Response(error_message, status=400)
    tables = database_handler.get_table_names_of_given_database(schema_name)
    if table_name not in tables:
        error_message = "encoding plan request's table name doesn't exist in the database"
        logger.error(error_message)
        return Response(error_message, status=400)
    time_base_field_name = request_parameter.get("time_base_field_name")
    if time_base_field_name is None or not time_base_field_name:
        error_message = "encoding plan request's time base field name is missing"
        logger.error(error_message)
        return Response(error_message, status=400)
    encoding_parameters = request_parameter.get("encoding_parameters")
    if encoding_parameters is None or not encoding_parameters:
        error_message = "encoding plan request's encoding parameters are missing"
        logger.error(error_message)
        return Response(error_message, status=400)
    feature_engineering_parameters = request_parameter.get("feature_engineering_parameters")
    if feature_engineering_parameters is None or not feature_engineering_parameters:
        logger.info("feature engineering didn't set")

    detailed_information_about_table = database_handler.get_detailed_information_about_table(schema_name=schema_name,
                                                                                             table_name=table_name)

    try:
        check_time_base_field_name(detailed_information_about_table, time_base_field_name)
        check_if_plan_exist_yet(schema_name, table_name, time_base_field_name, encoding_parameters,
                                feature_engineering_parameters)
        check_fields_and_possible_encoding_and_engineering(detailed_information_about_table,
                                                           encoding_parameters, feature_engineering_parameters)

    except NoTimeBaseFieldInTableException as e:
        details = e.args[0]
        message = details.get("message")
        logger.error(message)
        return Response(message, status=400)
    except FieldNotApplicableForTimeBaseException as e:
        details = e.args[0]
        message = details.get("message")
        field_name = details.get("field_name")
        error_message = f"{message}, field name: {field_name}"
        logger.error(error_message)
        return Response(error_message, status=400)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message")
        logger.error(message)
        return Response(message, status=400)
    except NoEnoughParameterException as e:
        details = e.args[0]
        message = details.get("message")
        parameter_name = details.get("parameter_name")
        error_message = f"{message}, parameter: {parameter_name}"
        logger.error(error_message)
        return Response(error_message, status=400)
    except FieldNotExistInTableException as e:
        details = e.args[0]
        message = details.get("message")
        field_name = details.get("field_name")
        error_message = f"{message}, schema name: {schema_name}, table name: {table_name}, field name: {field_name}"
        logger.error(error_message)
        return Response(error_message, status=400)
    except FieldNotApplicableForFraudException as e:
        details = e.args[0]
        message = details.get("message")
        field_name = details.get("field_name")
        error_message = f"{message}, schema name: {schema_name}, table name: {table_name}, field name: {field_name}"
        logger.error(error_message)
        return Response(error_message, status=400)
    except EncodingNotApplicableException as e:
        details = e.args[0]
        message = details.get("message")
        field_name = details.get("field_name")
        planned_encoding_type = details.get("planned_encoding_type")
        error_message = f"{message}, schema name: {schema_name}, table name: {table_name}, field name: {field_name}, planned encoding type: {planned_encoding_type}"
        logger.error(error_message)
        return Response(error_message, status=400)
    except NoFraudTypeFieldException as e:
        error_message = "There aren't any fraud type field in encoding parameters"
        logger.error(error_message)
        return Response(error_message, status=400)
    except TooManyFraudTypeFieldException as e:
        error_message = "There are too many fraud type field in encoding parameters"
        logger.error(error_message)
        return Response(error_message, status=400)
    except FeatureEngineeringNotApplicableException as e:
        details = e.args[0]
        message = details.get("message")
        field_name = details.get("field_name")
        field_type = details.get("field_type")
        error_message = f"{message}" + (f", field name: {field_name}" if field_type is not None else "") + (
            f", field type: {field_type}" if field_type is not None else "")
        logger.error(error_message)
        return Response(error_message, status=400)
    last_row_id = database_handler.persist_encoding_and_engineering_plan(schema_name, table_name,
                                                                         detailed_information_about_table,
                                                                         time_base_field_name,
                                                                         encoding_parameters,
                                                                         feature_engineering_parameters)
    return jsonify(last_row_id)


@app.route('/encoding_and_feature_engineering_plan', methods=['GET'])
def get_encoded_table():
    id = request.args.get("id")
    try:
        if id is None or id == "":
            result = database_handler.get_all_encoding_and_feature_engineering_plan()
        else:
            result = database_handler.get_encoding_and_feature_engineering_plan_by_id(id)
    except database_module.NoneException as e:
        details = e.args[0]
        message = details.get("message")
        parameter = details.get("parameter")
        error_message = f"{message}, id: {parameter}"
        logger.error(error_message)
        return Response(error_message, status=400)
    return jsonify(result)


@app.route('/available_feature_selector', methods=['GET'])
def get_available_feature_selector():
    result = dict()
    feature_selectors = get_feature_selectors()
    for key, value in feature_selectors.items():
        result[key] = value[0]
    return jsonify(result)


@app.route('/available_sampler', methods=['GET'])
def get_available_sampler():
    result = dict()
    samplers = get_samplers()
    for key, value in samplers.items():
        result[key] = value[0]
    return jsonify(result)


@app.route('/available_scaler', methods=['GET'])
def get_available_scaler():
    result = dict()
    scalers = get_scalers()
    for key, value in scalers.items():
        result[key] = value[0]
    return jsonify(result)


@app.route('/available_model', methods=['GET'])
def get_available_model():
    result = dict()
    models = get_models()
    for key, value in models.items():
        result[key] = value[0]
    return jsonify(result)


@app.route('/train_task', methods=['POST'])
def save_train_task():
    if not request.is_json:
        error_message = "request body must be JSON"
        logger.error(error_message)
        return Response(error_message, status=415)
    feature_selector_keys = get_feature_selectors().keys()
    sampler_keys = get_samplers().keys()
    scaler_keys = get_scalers().keys()
    model_keys = get_models().keys()
    request_parameter = request.get_json()
    if request_parameter is None:
        error_message = "Train task POST body is empty"
        logger.error(error_message)
        return Response(error_message, status=400)
    try:
        result = database_handler.persist_train_task(request_parameter, feature_selector_keys, sampler_keys,
                                                     scaler_keys, model_keys)
    except database_module.IllegalArgumentException as e:
        details = e.args[0]
        message = details.get("message")
        parameter_name = details.get("parameter_name")
        value = details.get("value")
        id = details.get("id")
        error_message = f"{message}" + (
            f", parameter name: {parameter_name}" if request_parameter is not None else "") + (
                            f", value: {value}" if value is not None else "") + (
                            f", id: {id}" if id is not None else "")
        logger.error(error_message)
        return Response(error_message, status=400)
    return jsonify(result)


@app.route('/train_task', methods=['GET'])
def get_train_task():
    id = request.args.get("id")
    if id is None:
        result = database_handler.get_all_train_task()
    else:
        result = database_handler.get_train_task_by_id(id)
    return jsonify(result)


@app.route('/fit', methods=['POST'])
def fit():
    if not request.is_json:
        error_message = "request body must be JSON"
        logger.error(error_message)
        return Response(error_message, status=415)

    request_parameter = request.get_json()
    train_task_id = request_parameter.get("train_task_id")
    used_cpu_core = request_parameter.get("used_cpu_core")
    available_cpu_core = os.cpu_count()

    train_task_status = database_handler.get_train_task_status(train_task_id)
    if train_task_status == IN_POGRESS:
        error_message = f"this train task is in progress yet, it can not be launched once more, train_task_id: {train_task_id}"
        logger.error(error_message)
        return Response(error_message, status=400)
    if train_task_status == FINISHED:
        error_message = f"this train task has been finished yet, train_task_id: {train_task_id}"
        logger.error(error_message)
        return Response(error_message, status=400)
    is_any_train_task_in_progress = database_handler.check_if_any_train_in_progress()
    if is_any_train_task_in_progress:
        error_message = "train in progress yet, it can not be launched another train task"
        logger.error(error_message)
        return Response(error_message, status=400)

    database_handler.set_train_task_status(train_task_id, IN_POGRESS)

    available_feature_selectors = get_feature_selectors(used_cpu_core)
    available_samplers = get_samplers(used_cpu_core)
    available_scalers = get_scalers()
    available_models = get_models(used_cpu_core)
    try:
        if used_cpu_core is None:
            message = f"used cpu core number not set, use the whole available logical cpu core, core number: {available_cpu_core}"
            logger.info(message)
            used_cpu_core = available_cpu_core
        else:
            if used_cpu_core > available_cpu_core:
                message = f"To many cpu core has been set, use the whole available logical cpu core, core number: {available_cpu_core}"
                logger.info(message)
                used_cpu_core = available_cpu_core
        if train_task_id is None:
            message = "train_task_id not set"
            raise InValidRequestParameterException({"message": message})

        train_parameters = database_handler.get_train_parameters(train_task_id)
        if train_parameters is None:
            message = "No train task belongs to this id"
            logger.error(message)
            raise InValidRequestParameterException({"message": message})
        planned_encoding_and_feature_engineering_id = train_parameters[1]
        if planned_encoding_and_feature_engineering_id is None:
            message = "Planned encoding and feature engineering id not set"
            raise InValidRequestParameterException({"message": message})

        feature_selector_key = train_parameters[2]
        if feature_selector_key is None:
            message = "Feature selector not set"
            raise InValidRequestParameterException({"message": message})

        feature_selector_name = available_feature_selectors.get(int(feature_selector_key))[0]
        feature_selector = available_feature_selectors.get(int(feature_selector_key))[1]
        logger.info(f"Feature selector: {feature_selector_name}")
        if feature_selector is None:
            message = "Feature selector doesn't exist in train application"
            raise InValidRequestParameterException({"message": message})

        sampler_key = train_parameters[3]
        if sampler_key is None:
            message = "Sampler not set"
            raise InValidRequestParameterException({"message": message})
        sampler_name = available_samplers.get(int(sampler_key))[0]
        sampler = available_samplers.get(int(sampler_key))[1]
        logger.info(f"Sampler: {sampler_name}")
        if sampler is None:
            message = "Sampler doesn't exist in train application"
            raise InValidRequestParameterException({"message": message})

        scaler_key = train_parameters[4]
        if scaler_key is None:
            message = "Scaler not set"
            raise InValidRequestParameterException({"message": message})
        scaler_name = available_scalers.get(int(scaler_key))[0]
        scaler = available_scalers.get(int(scaler_key))[1]
        logger.info(f"Scaler: {scaler_name}")
        if scaler is None:
            message = "Scaler doesn't exist in train application"
            raise InValidRequestParameterException({"message": message})

        model_key = train_parameters[5]
        if model_key is None:
            message = "Model not set"
            raise InValidRequestParameterException({"message": message})
        model_name = available_models.get(int(model_key))[0]
        model = available_models.get(int(model_key))[1]
        logger.info(f"Model: {model_name}")
        if model is None:
            message = "Model doesn't exist in train application"
            raise InValidRequestParameterException({"message": message})

        split_test_size = None
        saved_test_split_size = train_parameters[6]
        if saved_test_split_size is None:
            logger.info("No test split size set, use default value, 0.25")
        else:
            split_test_size = saved_test_split_size
            logger.info(f"Train test split size: {split_test_size}")

        fit_process_parameter = dict()
        fit_process_parameter["train_task_id"] = train_task_id
        fit_process_parameter[
            "planned_encoding_and_feature_engineering_id"] = planned_encoding_and_feature_engineering_id
        fit_process_parameter["feature_selector"] = feature_selector
        fit_process_parameter["sampler"] = sampler
        fit_process_parameter["scaler"] = scaler
        fit_process_parameter["model"] = model
        fit_process_parameter["used_cpu_core"] = used_cpu_core
        fit_process_parameter["split_test_size"] = split_test_size
        thread = Thread(target=process_fit, args=(fit_process_parameter,))
        thread.start()
        logger.info(f"Train parameters are proper, fit process started, train_task_id: {train_task_id}")
        response = Response(status=200)
    except InValidRequestParameterException as e:
        details = e.args[0]
        message = details.get("message")
        logger.error(message)
        request_message = message + (
            f", train_task_id: {train_task_id}" if train_task_id is not None else "")
        response = Response(request_message, status=400)
    return response


@app.route('/metrics', methods=['GET'])
def get_metrics():
    estimator_id = request.args.get("estimator_id")
    if estimator_id is None:
        result = database_handler.get_all_metrics()
    else:
        result = database_handler.get_metrics_by_estimator_id(estimator_id)
    return jsonify(result)


@app.route('/estimator', methods=['GET'])
def get_estimators():
    train_task_id = request.args.get("train_task_id")
    if train_task_id is None:
        result = database_handler.get_all_estimator()
    else:
        result = database_handler.get_estimator_by_train_task_id(train_task_id)
    return jsonify(result)


def check_time_base_field_name(detailed_information_about_table, time_base_field_name):
    field_names = list()
    time_base_field_names = list()
    field_properties = detailed_information_about_table.get("fields")
    for field_property in field_properties:
        field_type = field_property.get("type")
        field_name = field_property.get("name")
        field_names.append(field_name)
        if field_type == "datetime":
            time_base_field_names.append(field_name)
    if time_base_field_name not in field_names:
        raise FieldNotExistInTableException(
            {"message": "The field doesn't exist in this database", "field_name": time_base_field_name})
    if not time_base_field_names:
        raise NoTimeBaseFieldInTableException({"message": "There isn't any time base field in the table"})
    if not time_base_field_name in time_base_field_names:
        raise FieldNotApplicableForTimeBaseException(
            {"message": "The specified field unsuitable for time base in the table",
             "field_name": time_base_field_name})


def check_if_plan_exist_yet(schema_name, table_name, time_base_field_name, encoding_parameters,
                            feature_engineering_parameters):
    existing_plans = database_handler.get_all_encoding_and_feature_engineering_plan()
    for plan_key in existing_plans.keys():
        plan = existing_plans.get(plan_key)
        if plan.get("schema_name") == schema_name and plan.get("table_name") == table_name and plan.get(
                "encoding_parameters") == encoding_parameters and plan.get(
            "feature_engineering_parameters") == feature_engineering_parameters and plan.get(
            "time_base_field_name") == time_base_field_name:
            message = "Encoding and feature engineering plan exist yet"
            raise InValidRequestParameterException({"message": message})


def check_fields_and_possible_encoding_and_engineering(detailed_information_about_table,
                                                       encoding_parameters, feature_engineering_parameters):
    field_properties = detailed_information_about_table.get("fields")
    number_of_int_and_float_type_fields = get_number_of_int_and_float_typed_fields(field_properties)
    field_names_as_keys_in_encoding_parameters = encoding_parameters.keys()
    number_of_encoding_parameters = len(field_names_as_keys_in_encoding_parameters)
    number_of_fields_in_current_table = len(field_properties)
    number_of_fields_to_be_encoded = number_of_fields_in_current_table - number_of_int_and_float_type_fields - 2  # fraud and id
    if number_of_encoding_parameters < number_of_fields_to_be_encoded:
        raise NoEnoughParameterException(
            {"message": "there aren't enough parameter to encode", "parameter_name": encoding_parameters})
    field_names_in_table = get_field_names_from_field_and_type_properties(field_properties)
    for field in field_names_as_keys_in_encoding_parameters:
        if field not in field_names_in_table:
            raise FieldNotExistInTableException(
                {"message": "The field doesn't exist in this database", "field_name": field})
        if encoding_parameters.get(field) == "fraud_type":
            check_if_field_applicable_for_fraud(detailed_information_about_table, field)

    all_field_names = check_encoding_parameters(encoding_parameters,
                                                detailed_information_about_table)
    check_number_of_fraud_type_field(all_field_names, encoding_parameters)
    if feature_engineering_parameters is not None:
        check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                feature_engineering_parameters)


def check_encoding_parameters(encoding_parameters, detailed_information_about_table):
    all_field_names = list()
    field_properties = detailed_information_about_table.get("fields")
    for field_property in field_properties:
        field_name = field_property.get("name")
        all_field_names.append(field_name)
        if field_name in encoding_parameters.keys():
            planned_encoding_type = encoding_parameters.get(field_name)
            if (planned_encoding_type == "julian" and field_property.get(
                    "type") != "datetime") or (planned_encoding_type == "label_encoder" and field_property.get(
                "type") != "varchar") or (planned_encoding_type == "int" and "int" not in field_property.get(
                database_module.OTHER_APPLICABLE_ENCODING_TYPES)) or (
                    planned_encoding_type == "float" and "float" not in field_property.get(
                database_module.OTHER_APPLICABLE_ENCODING_TYPES)) or planned_encoding_type not in ["julian",
                                                                                                   "label_encoder",
                                                                                                   "int",
                                                                                                   "float",
                                                                                                   "fraud_type",
                                                                                                   "omit"]:
                raise EncodingNotApplicableException(
                    {"message": "the planned encoding can not applicable to this field", "field_name": field_name,
                     "planned_encoding_type": planned_encoding_type})
    return all_field_names


def get_number_of_int_and_float_typed_fields(field_and_type_and_other_eligible_type_for_encoding_collection):
    int_types = ("tinyint", "smallint", "mediumint", "int", "bigint")
    float_types = ("float", "double")
    number_of_int_and_float_typed_field = 0
    for field_property in field_and_type_and_other_eligible_type_for_encoding_collection:
        current_type = field_property.get("type")
        if current_type in int_types or current_type in float_types:
            number_of_int_and_float_typed_field += 1
    return number_of_int_and_float_typed_field


def check_if_field_applicable_for_fraud(detailed_information_about_table, current_field_name):
    fraud_candidates = detailed_information_about_table.get("fraud_candidates")
    applicable_field_names_for_fraud = list()
    for fraud_candidate in fraud_candidates:
        applicable_field_names_for_fraud.append(fraud_candidate.get("name"))
    if current_field_name not in applicable_field_names_for_fraud:
        raise FieldNotApplicableForFraudException(
            {"message": "the field not applicable for fraud", "field_name": current_field_name})


def check_number_of_fraud_type_field(field_names, encoding_parameters):
    number_of_fraud_type_field = 0
    for field_name in field_names:
        if encoding_parameters.get(field_name) == "fraud_type":
            number_of_fraud_type_field += 1
    if number_of_fraud_type_field == 0:
        raise NoFraudTypeFieldException()
    if number_of_fraud_type_field > 1:
        raise TooManyFraudTypeFieldException()


def check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                            feature_engineering_parameters):
    field_name_to_be_engineered = feature_engineering_parameters.get("chosen_feature_field")
    intervals = feature_engineering_parameters.get("intervals")
    if field_name_to_be_engineered is None:
        message = "field name for engineering is missing"
        raise FeatureEngineeringNotApplicableException({"message": message})
    if intervals is None or not intervals:
        message = "interval list for engineering is missing or empty"
        raise FeatureEngineeringNotApplicableException({"message": message})
    field_properties = detailed_information_about_table.get("fields")
    field_names = get_field_names_from_field_and_type_properties(field_properties)
    if field_name_to_be_engineered not in field_names:
        message = "the field name to be engineered doesn't exist in the table"
        logger.error(message + f", field name to be engineered: {field_name_to_be_engineered}")
        raise FeatureEngineeringNotApplicableException({"message": message})
    for field_property in field_properties:
        field_name = field_property.get("name")
        field_type = field_property.get("type")
        if field_name == field_name_to_be_engineered:
            if field_type not in ["int", "float"]:
                message = "only int or float type field can be used for feature engineering"
                raise FeatureEngineeringNotApplicableException(
                    {"message": message, "field name": field_name, "field's type": field_type})
            if encoding_parameters.get(field_name) == "omit":
                message = f"the {field_name} field has been omitted during encoding process, it can't be used for feature engineering"
                raise FeatureEngineeringNotApplicableException({"message": message})
    if len(intervals) == 0:
        message = "time intervals list is empty"
        raise FeatureEngineeringNotApplicableException({"message": message})


def get_field_names_from_field_and_type_properties(field_and_type_properties):
    result = list()
    for item in field_and_type_properties:
        result.append(item.get("name"))
    return result


def is_int(element):
    try:
        int(element)
        return True
    except ValueError:
        return False


def process_fit(parameter):
    message = dict()
    train_task_id = parameter.get("train_task_id")
    logger.info(f"train process started, train_task_id: {train_task_id}")
    planned_encoding_and_feature_engineering_id = parameter.get("planned_encoding_and_feature_engineering_id")
    schema_name, table_name, detailed_information_about_table, time_base_field_name, encoding_parameters, feature_engineering_parameters = database_handler.get_parameters_from_plan_by_id(
        planned_encoding_and_feature_engineering_id)
    feature_selector = parameter.get("feature_selector")
    sampler = parameter.get("sampler")
    scaler = parameter.get("scaler")
    model = parameter.get("model")
    split_test_size = parameter.get("split_test_size")
    used_cpu_core = parameter.get("used_cpu_core")

    raw_dataset_id = database_handler.get_raw_dataset_id(schema_name, table_name)
    encoding_id = database_handler.get_encoding_id_by_json_content(encoding_parameters)

    encoded_table_registry_id, encoded_table_name, encoded_field_names, label_encoder_registry = database_handler.get_encoded_table_properties(
        raw_dataset_id, encoding_id, time_base_field_name)
    if encoded_table_name is not None:
        logger.info(f"database encoded yet, train task id: {train_task_id}")
        send_async_message(train_task_id, DATABASE_ENCODED_YET)
        dataset_to_train = database_handler.get_records_for_processing(schema_name, encoded_table_name,
                                                                       time_base_field_name)
    else:
        start_encoding = time.time()
        logger.info(f"database encoding started, train task id: {train_task_id}")
        send_async_message(train_task_id, DATABASE_ENCODING_BEGAN)
        try:
            dataset_to_train, encoded_table_registry_id, encoded_table_name, encoded_field_names, label_encoder_registry = database_handler.encode(
                raw_dataset_id, schema_name, table_name, encoding_id, encoding_parameters, time_base_field_name)
        except Exception as e:
            logger.error("Error occured: " + str(e))

        finish_encoding = time.time()
        encoding_process_time = finish_encoding - start_encoding
        logger.info(
            f"database encoding finished, train task id: {train_task_id}, processing time: {encoding_process_time} sec")
        send_async_message(train_task_id, DATABASE_ENCODING_FINISHED)

    feature_engineered_insert_script_using_in_prediction = None
    if feature_engineering_parameters is None:
        logger.info("train without feature engineering")
    else:
        feature_engineering_id = database_handler.get_feature_engineering_id_by_json_content(
            feature_engineering_parameters)
        table_name, feature_engineered_insert_script = database_handler.get_table_name_from_feature_engineering_registry(
            encoded_table_registry_id,
            feature_engineering_id)
        if table_name is not None:
            logger.info(f"database feature engineered yet, train task id: {train_task_id}")
            dataset_to_train = database_handler.get_records_for_processing(schema_name, table_name,
                                                                           time_base_field_name)
        else:
            start_feature_engineering = time.time()
            logger.info(f"database feature engineering started, train task id: {train_task_id}")
            send_async_message(train_task_id, DATABASE_FEATURE_ENGINEERED_BEGAN)
            fraud_type_field_name = get_fraud_type_field_name(encoding_parameters)
            dataset_to_train, table_name, feature_engineered_insert_script_using_in_prediction = database_handler.feature_engineer(
                schema_name, encoded_table_name, fraud_type_field_name,
                feature_engineering_id,
                dataset_to_train, time_base_field_name,
                encoded_field_names,
                feature_engineering_parameters,
                encoded_table_registry_id, used_cpu_core)
            finish_feature_engineering = time.time()
            feature_engineering_process_time = finish_feature_engineering - start_feature_engineering
            logger.info(
                f"database feature engineering finished, train task id: {train_task_id}, processing time: {feature_engineering_process_time} sec")
            send_async_message(train_task_id, DATABASE_FEATURE_ENGINEERED_FINISHED)

    features = dataset_to_train[:, :-1]
    labels = dataset_to_train[:, -1:]
    labels = labels.astype(int)
    labels = column_or_1d(labels)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=split_test_size,
                                                                                random_state=0)

    logger.info("sampling started")
    send_async_message(train_task_id, SAMPLING_BEGAN)
    start_sampling = time.time()
    sampled_train_features, sampled_train_labels = sampler.fit_resample(train_features, train_labels)
    finish_sampling = time.time()
    sampling_proessing_time = finish_sampling - start_sampling
    logger.info(
        f"sampling finished, train task id: {train_task_id}, processing time: {sampling_proessing_time} sec")
    send_async_message(train_task_id, SAMPLING_FINISHED)

    pipeline = Pipeline(
        [('scaler', scaler), ('featureSelector', feature_selector), ('model', model)]
    )
    logger.info(f"pipeline's training started, train task id: {train_task_id}")
    send_async_message(train_task_id, PIPELINE_FIT_BEGAN)
    start_fit = time.time()
    pipeline.fit(sampled_train_features, sampled_train_labels)
    finish_fit = time.time()
    fit_processing_time = finish_fit - start_fit
    logger.info(
        f"pipeline's training finished, train task id: {train_task_id} ,processing time: {fit_processing_time} sec")
    send_async_message(train_task_id, PIPELINE_FIT_FINISHED)
    predicted_labels = pipeline.predict(test_features)
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    logger.info(f"Confusion Matrix: {conf_matrix}, train_task_id: {train_task_id}")
    estimator = dict()

    estimator["pipeline"] = pipeline
    estimator["label_encoder_registry"] = label_encoder_registry
    estimator["encoding_parameters"] = encoding_parameters
    estimator["feature_engineering_parameters"] = feature_engineering_parameters
    estimator["encoded_field_names"] = encoded_field_names
    estimator["time_base_field_name"] = time_base_field_name
    estimator["schema_name"] = schema_name
    estimator["feature_engineered_table_name"] = table_name
    estimator["detailed_information_about_table"] = detailed_information_about_table
    estimator["feature_engineered_insert_script"] = feature_engineered_insert_script_using_in_prediction

    estimator_id = database_handler.persist_estimator(train_task_id, estimator)
    calculate_and_save_metrics(estimator_id, conf_matrix, test_labels, predicted_labels)
    logger.info(
        f"estimator (pipeline, label encoders, encoding parameters, feature engineering parameters etc.) persisted, train task id: {train_task_id}, estimator_id: {estimator_id}")
    message["method"] = "PIPELINE_PERSISTED"
    send_async_message(train_task_id, PIPELINE_PERSISTED)
    database_handler.set_train_task_status(train_task_id, FINISHED)
    logger.info(f"train process finished, train_task_id: {train_task_id}")


def get_fraud_type_field_name(encoding_parameters):
    for field_name, encoding_type in encoding_parameters.items():
        if encoding_type == FRAUD_TYPE:
            return field_name


def calculate_and_save_metrics(estimator_id, conf_matrix, test_labels, predicted_labels):
    TP = int(conf_matrix[0][0])
    FP = int(conf_matrix[0][1])
    FN = int(conf_matrix[1][0])
    TN = int(conf_matrix[1][1])
    temp = TP + FN
    sensitivity = 0
    if temp != 0:
        sensitivity = TP / (TP + FN)
    temp = TN + FP
    specificity = 0
    if temp != 0:
        specificity = TN / (TN + FP)
    accuracy = accuracy_score(test_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(test_labels, predicted_labels)
    precision = 0
    temp = TP + FP
    if temp != 0:
        precision = TP / (TP + FP)
    recall = recall_score(test_labels, predicted_labels)
    temp = TP + FN
    PPV = 0
    if temp != 0:
        PPV = TP / (TP + FN)
    temp = TN + FN
    NPV = 0
    if temp != 0:
        NPV = TN / (TN + FN)
    temp = FN + TP
    FNR = 0
    if temp != 0:
        FNR = FN / (FN + TP)
    temp = FP + TN
    FPR = 0
    if temp != 0:
        FPR = FP / (FP + TN)
    FDR = 0
    temp = FP + TP
    if temp != 0:
        FDR = FP / (FP + TP)
    temp = FN + TN
    FOR = 0
    if temp != 0:
        FOR = FN / (FN + TN)
    f1 = f1_score(test_labels, predicted_labels)
    f_05 = calculateF(0.5, precision, recall)
    f2 = calculateF(2, precision, recall)
    temp = math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN)
    MCC = 0
    if temp != 0:
        MCC = (TP * TN - FP * FN) / temp
    ROCAUC = roc_auc_score(test_labels, predicted_labels)
    Youdens_statistic = sensitivity + specificity - 1
    database_handler.persist_metrics(estimator_id, TP, FP, TN, FN, sensitivity, specificity, accuracy,
                                     balanced_accuracy,
                                     precision, recall, PPV, NPV, FNR, FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC,
                                     Youdens_statistic)


def calculateF(beta, precision, recall):
    temp = beta * beta * precision + recall
    if temp != 0:
        f_beta = (1 + beta) * (1 + beta) * precision * recall / temp
    else:
        f_beta = 0
    return f_beta


if __name__ == "__main__":
    log_file = os.getenv("TRAIN_LOG_FILE", "/home/tomi/log/train.log")
    log_level = os.getenv("TRAIN_LOG_LEVEL", "DEBUG")
    rabbitmq_url = os.getenv("RABBITMQ_URL", "localhost")
    database_url = os.getenv("MYSQL_DATABASE_URL", "localhost")
    database_user = os.getenv("MYSQL_USER", "root")
    database_password = os.getenv("MYSQL_PASSWORD", "pwd")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger("train.server")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.debug("RabbitMQ: " + rabbitmq_url)
    logger.debug("MYSQL_DATABASE_URL: " + database_url)
    logger.debug("MYSQL_USER :" + database_user)
    logger.debug("MYSQL_PASSWORD: " + database_password)
    logger.debug("Train application started")
    database_handler = database_module.Handler(database_url, database_user, database_password)
    database_handler.create_common_fraud_schemas()
    logger.debug("train modul started")
    app.run(host='0.0.0.0', port=8085, debug=True)
