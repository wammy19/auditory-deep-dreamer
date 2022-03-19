import os
import pickle
import re
from csv import DictWriter
from os.path import exists, join
from typing import Callable, Dict, List, Optional, Tuple, Union

from aim.keras import AimCallback
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.models import Model, load_model

from ai_tools import DataGenerator


class ModelManager:
    """
    Creates, trains, and evaluates. Logs are kept of the models created as well as their performance. The best
    performing model epochs are saved.
    """


    # =================================================================================================================
    # ---------------------------------------------- Class Constructors -----------------------------------------------
    # =================================================================================================================

    def __init__(
            self,
            model_builder_func: Callable,
            train_data_generator: DataGenerator,
            validation_data_generator: DataGenerator,
            test_data_generator: DataGenerator,
            path_to_logs: str = './logs',
            model_checkpoint_dir: str = './models',
            training_batch_size: int = 32,
            num_training_epochs: int = 200,
            model: Model = None
    ):
        """
        :param path_to_logs:
        :param model_checkpoint_dir:
        :param training_batch_size:
        :param num_training_epochs:
        :param model:
        """

        # Logs paths.
        self._path_to_model_evaluation_logs: str = join(path_to_logs, 'model_evaluation.csv')
        self._aim_logs_dir: str = join(path_to_logs, 'aim')
        self._history_log_dir: str = join(path_to_logs, 'model_histories')
        self._path_to_logs: str = path_to_logs
        self._model_checkpoint_dir: str = model_checkpoint_dir

        # Logs.
        self._current_history: Union[History, None] = None
        self._current_model_settings: Union[Dict[str, any], None] = None
        self._current_model_builder: Union[str, None] = None

        self._verify_log_dirs_and_files_exist()

        # Model building.
        self.model_builder: Callable = model_builder_func

        # Datasets.
        self.train_data_generator: DataGenerator = train_data_generator
        self.validation_data_generator: DataGenerator = validation_data_generator
        self.test_data_generator: DataGenerator = test_data_generator

        # Initialize model ID.
        self._model_ID: int = len(os.listdir(model_checkpoint_dir))

        # Training settings.
        self.num_epochs: int = num_training_epochs
        self._batch_size: int = training_batch_size
        self._current_model: Optional[Model] = model

        # Patterns
        self._epoch_from_model_logs_patter: re.Pattern = re.compile(r'\d+')


    # =================================================================================================================
    # ----------------------------------------------- Public functions ------------------------------------------------
    # =================================================================================================================

    def search_for_best_model(self, epochs: int = 100, early_stopping_patience: int = 5, **kwargs) -> float:

        self.current_model = self._build_model(**kwargs)

        self._train_model(
            self.current_model,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience
        )

        # Load best model at best epoch for evaluation. None will be returned if the name can't be found in the logs.
        model, best_epoch = self.load_model_at_best_epoch(self._model_ID)  # type: Model, str
        results: List[float] = model.evaluate(self.test_data_generator)

        # Log model settings, evaluation and training history.
        self._save_model_history(self._current_history)
        self._save_model_settings_to_csv(self._current_model_builder, self._current_model_settings)
        self._save_model_evaluation_to_csv(self._model_ID, results[0], results[1], best_epoch)

        self._model_ID += 1

        return results[1]  # Return model accuracy.


    def get_model_history(self, model_id: Optional[int] = None) -> Union[History, None]:
        """
        :param model_id: Model ID number of which you wish to get the training history of.
        :return:

        Returns the history of a previously trained model if the model ID is provided, otherwise returns the current
        history stored.
        """

        if model_id is not None:  # Return History with a provided model_id.
            try:
                with open(join(self._history_log_dir, f'model_{model_id}'), 'rb') as file_handler:
                    history: History = pickle.load(file_handler)

                    print(f'Got history for model_{model_id}')

                    return history

            except FileNotFoundError:
                print("Model ID provided doesn't exist. Please check logs directory for existing logs.")

        else:  # Return History of the current model loaded.

            if self._current_history is None:
                print(
                    'There is currently no history stored. '
                    'Build and train a model before trying to get the current History.'
                )

                return None

            else:
                print(f'Got history for last model train. model_{self._model_ID}')
                return self._current_history


    def load_model_at_best_epoch(self, model_id: int) -> Tuple[Union[Model, None], Union[str, None]]:
        """
        :param model_id:
        :return:
        """

        try:
            path_to_model: str = join(self._model_checkpoint_dir, f'model_{model_id}')
            best_epoch_file_name: str = sorted(os.listdir(path_to_model))[-1]

            model: Model = load_model(join(path_to_model, best_epoch_file_name))
            epoch: str = self._epoch_from_model_logs_patter.findall(best_epoch_file_name)[0]

            return model, epoch

        except ValueError:
            print('Model ID provided does not correspond to any models saved.')

            return None, None


    # =================================================================================================================
    # ----------------------------------------------- Private functions -----------------------------------------------
    # =================================================================================================================

    def _build_model(self, **kwargs) -> Model:
        """
        :param kwargs:
        :return:
        """

        # Store settings of model for later logging.
        self._current_model_builder = self.model_builder.__name__
        self._current_model_settings = kwargs

        model: Model = self.model_builder(**kwargs)

        return model


    def _train_model(
            self,
            model: Model,
            epochs: int = 100,
            early_stopping_patience: int = 5,
            update_current_model_id: bool = False
    ) -> None:
        """
        :param model: A compiled model ready for training.
        :param epochs: Number of epochs to train for, Early stopping is also in place.
        :param early_stopping_patience: EarlyStopping callback patience amount. This will stop training early if there
        is no improvement.
        :param update_current_model_id:
        :return:
        """

        # Callbacks.
        aim_callback = AimCallback(repo=self._aim_logs_dir, experiment=f'model_{self._model_ID}')
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=False)
        checkpoint = ModelCheckpoint(
            join(join(self._model_checkpoint_dir, f'model_{self._model_ID}'), 'epoch-{epoch:02d}.pb'),
            monitor='val_accuracy',
            verbose=False,
            save_weights_only=False,
            save_best_only=True,
            mode='max',
        )

        # Train model.
        self._current_history: History = model.fit(
            self.train_data_generator,
            steps_per_epoch=len(self.train_data_generator.get_data_frame.index) // self._batch_size,
            epochs=epochs,
            validation_data=self.validation_data_generator,
            validation_steps=len(self.validation_data_generator.get_data_frame.index) // self._batch_size,
            batch_size=self._batch_size,
            verbose=False,
            callbacks=[
                aim_callback,
                early_stopping,
                checkpoint
            ]
        )

        if update_current_model_id:
            self._model_ID += 1  # Increment model ID for the next model.


    def _save_model_settings_to_csv(self, model_name: str, model_config: Dict[str, any]) -> None:
        """
        :param model_config: A dict containing the parameter to constructing a model, and it's value.
        :return:

        Appends the model settings to a csv file.
        """

        new_model_config: Dict[str, any] = {'model_ID': self._model_ID}
        new_model_config.update(model_config)

        csv_headers: List[str] = []

        for key, value in new_model_config.items():  # Gather column headers.
            csv_headers.append(key)

        path_to_csv: str = join(self._path_to_logs, f'{model_name}.csv')

        with open(path_to_csv, 'a', newline='') as file_handler:
            dict_writer = DictWriter(file_handler, fieldnames=csv_headers)

            # Write column headers if the file is new.
            if os.stat(path_to_csv).st_size == 0:
                dict_writer.writeheader()

            dict_writer.writerow(new_model_config)


    def _save_model_evaluation_to_csv(
            self,
            model_id: int,
            model_loss: float,
            model_accuracy: float,
            epoch: str
    ) -> None:
        """
        :param model_id: Model id number.
        :param model_loss:
        :param model_accuracy:
        :param epoch: Epoch the results were measured at.
        :return:
        """

        evaluation_res: Dict[str, any] = {
            'model_ID': model_id,
            'loss': model_loss,
            'accuracy': model_accuracy,
            'epoch': epoch
        }

        csv_headers: List[str] = []

        for key, value in evaluation_res.items():  # Gather column headers.
            csv_headers.append(key)

        with open(self._path_to_model_evaluation_logs, 'a', newline='') as file_handler:
            dict_writer = DictWriter(file_handler, fieldnames=csv_headers)

            # Write column headers if the file is new.
            if os.stat(self._path_to_model_evaluation_logs).st_size == 0:
                dict_writer.writeheader()

            dict_writer.writerow(evaluation_res)


    def _save_model_history(self, history: History) -> None:
        """
        :param history: A tensorflow.keras.callbacks.History object.
        :return:

        Pickles the model history object for later plotting.
        """

        with open(join(self._history_log_dir, f'model_{self._model_ID}'), 'wb') as file_handler:
            pickle.dump(history, file_handler)


    def _verify_log_dirs_and_files_exist(self) -> None:
        """
        :return: None

        Creates log directories and files if they don't exist.
        """

        if exists(self._path_to_model_evaluation_logs) is False:
            open(self._path_to_model_evaluation_logs, 'x')

        if exists(self._model_checkpoint_dir) is False:
            os.makedirs(self._model_checkpoint_dir)

        if exists(self._aim_logs_dir) is False:
            os.makedirs(self._aim_logs_dir)

        if exists(self._history_log_dir) is False:
            os.makedirs(self._history_log_dir)


    # =================================================================================================================
    # ----------------------------------------------- Getter/Setter functions -----------------------------------------
    # =================================================================================================================

    @property
    def current_model(self) -> Union[Model, None]:
        """
        :return: A tensorflow.keras.models.Model or None if a model hasn't been set.

        Returns the last model that was constructed by the model manager. None will be returned if no model has
        been built yet.
        """

        return self._current_model


    @current_model.setter
    def current_model(self, model: Model) -> None:
        """
        :param model: a compiled tensorflow.keras.models.Model ready for fitting.
        :return:

        Updates the model stored by the manager with a model that is passed in.
        """

        # Type check.
        if isinstance(model, Model) is False:
            raise ValueError('The current model must be a compiled tensorflow.keras.models.Model')

        self._current_model = model
