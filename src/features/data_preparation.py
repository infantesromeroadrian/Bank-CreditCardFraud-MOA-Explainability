from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import logging
from src.utils.decorators import timer_decorator, error_handler, log_decorator

logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, data):
        self.data = data.copy()
        self.numeric_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        self.target_column = 'Class'

    @timer_decorator
    @error_handler
    @log_decorator
    def remove_irrelevant_columns(self):
        columns_to_drop = ['Time']
        existing_columns = [col for col in columns_to_drop if col in self.data.columns]
        if existing_columns:
            self.data.drop(columns=existing_columns, inplace=True)
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def handle_missing_values(self):
        num_imputer = SimpleImputer(strategy='mean')
        self.data[self.numeric_columns] = num_imputer.fit_transform(self.data[self.numeric_columns])
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def normalize_numeric_features(self):
        scaler = StandardScaler()
        self.data[self.numeric_columns] = scaler.fit_transform(self.data[self.numeric_columns])
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def balance_data_with_smote(self, test_size=0.3, random_state=42):
        X = self.data[self.numeric_columns]
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f'Datos después de SMOTE: {X_train_resampled.shape}, Clase 0: {sum(y_train_resampled == 0)}, Clase 1: {sum(y_train_resampled == 1)}')
        return X_train_resampled, X_test, y_train_resampled, y_test

    @timer_decorator
    @error_handler
    @log_decorator
    def prepare_data(self):
        return (self.remove_irrelevant_columns()
                    .handle_missing_values()
                    .normalize_numeric_features())

    @timer_decorator
    @error_handler
    @log_decorator
    def get_prepared_data(self):
        return self.data
