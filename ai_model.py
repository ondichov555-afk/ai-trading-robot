"""
AI Model Module - Machine Learning models for price prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import config


class AITradingModel:
    def __init__(self, model_type=config.MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM neural network model
        """
        model = Sequential([
            LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification: UP or DOWN
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self, input_shape):
        """
        Build GRU neural network model
        """
        from tensorflow.keras.layers import GRU
        
        model = Sequential([
            GRU(128, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            GRU(64, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_random_forest_model(self):
        """
        Build Random Forest model
        """
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the AI model
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-config.TRAIN_TEST_SPLIT, random_state=42
            )
            
            if self.model_type == "LSTM":
                self.model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
                # Normalize data
                X_train_scaled = X_train.reshape(-1, X_train.shape[-1])
                X_train_scaled = self.scaler.fit_transform(X_train_scaled)
                X_train_scaled = X_train_scaled.reshape(X_train.shape)
                
                X_test_scaled = X_test.reshape(-1, X_test.shape[-1])
                X_test_scaled = self.scaler.transform(X_test_scaled)
                X_test_scaled = X_test_scaled.reshape(X_test.shape)
                
                # Train model
                history = self.model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=1
                )
                
                # Evaluate
                test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test)
                print(f"✓ LSTM Model trained - Test Accuracy: {test_accuracy:.4f}")
                
            elif self.model_type == "GRU":
                self.model = self.build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                
                X_train_scaled = X_train.reshape(-1, X_train.shape[-1])
                X_train_scaled = self.scaler.fit_transform(X_train_scaled)
                X_train_scaled = X_train_scaled.reshape(X_train.shape)
                
                X_test_scaled = X_test.reshape(-1, X_test.shape[-1])
                X_test_scaled = self.scaler.transform(X_test_scaled)
                X_test_scaled = X_test_scaled.reshape(X_test.shape)
                
                history = self.model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=1
                )
                
                test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test)
                print(f"✓ GRU Model trained - Test Accuracy: {test_accuracy:.4f}")
                
            elif self.model_type == "RandomForest":
                self.model = self.build_random_forest_model()
                
                # Reshape data for Random Forest
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                
                self.model.fit(X_train_flat, y_train)
                
                train_accuracy = self.model.score(X_train_flat, y_train)
                test_accuracy = self.model.score(X_test_flat, y_test)
                print(f"✓ RandomForest Model trained - Test Accuracy: {test_accuracy:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"✗ Error training model: {str(e)}")
            return False
    
    def predict(self, X):
        """
        Make predictions with the model
        Returns confidence score (0-1)
        """
        if not self.is_trained or self.model is None:
            print("✗ Model not trained yet")
            return None
        
        try:
            if self.model_type in ["LSTM", "GRU"]:
                X_scaled = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_scaled)
                X_scaled = X_scaled.reshape(X.shape)
                prediction = self.model.predict(X_scaled, verbose=0)[0][0]
            else:  # RandomForest
                X_flat = X.reshape(X.shape[0], -1)
                prediction = self.model.predict_proba(X_flat)[0][1]
            
            return float(prediction)
        
        except Exception as e:
            print(f"✗ Error making prediction: {str(e)}")
            return None
    
    def save_model(self, filename="models/trading_model.h5"):
        """
        Save the trained model
        """
        try:
            if self.model_type in ["LSTM", "GRU"]:
                self.model.save(filename)
            else:
                joblib.dump(self.model, filename)
            print(f"✓ Model saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
    
    def load_model(self, filename="models/trading_model.h5"):
        """
        Load a trained model
        """
        try:
            if self.model_type in ["LSTM", "GRU"]:
                self.model = keras.models.load_model(filename)
            else:
                self.model = joblib.load(filename)
            self.is_trained = True
            print(f"✓ Model loaded from {filename}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")


# Example usage
if __name__ == "__main__":
    model = AITradingModel(model_type="LSTM")
    print(f"Model type: {model.model_type}")
    print("Model ready for training")
