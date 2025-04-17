import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

class MultinomialLogisticRegression:
    """
    Multinomial Logistic Regression
    
    Parameters:
    -----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float
        Smaller values specify stronger regularization.
    
    max_iter : int, default=100
        Maximum number of iterations for optimizer to converge
    
    learning_rate : float, default=0.1
        Initial learning rate for gradient descent
        
    tol : float, default=1e-4
        Tolerance for stopping criteria
        
    random_state : int, default=None
        Controls randomness for weight initialization and shuffling
        
    verbose : bool, default=False
        Whether to print progress during fitting
    """
    
    def __init__(self, C=1.0, max_iter=100, learning_rate=0.1, 
                 tol=1e-4, random_state=None, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        self.classes_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []
        
    def _softmax(self, z):
        """
        Compute softmax function with numerical stability
        """
        # Shift inputs for numerical stability (subtract max value)
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _add_intercept(self, X):
        """
        Add intercept term to feature matrix X
        """
        n_samples = X.shape[0]
        intercept = np.ones((n_samples, 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _initialize_params(self, n_features, n_classes):
        """
        Initialize parameters with appropriate scaling
        """
        np.random.seed(self.random_state)
        
        # Initialize with small random values scaled by sqrt(n_features)
        scale = 0.01 / np.sqrt(n_features + 1)  # Adjust scaling factor
        theta = np.random.randn(n_classes, n_features + 1) * scale
        
        return theta
    
    def _compute_loss(self, X, y, theta=None):
        """
        Compute cross-entropy loss with L2 regularization.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features + 1)
            Input data with intercept.
        
        y : array, shape (n_samples, n_classes)
            One-hot encoded target values.
            
        theta : array, shape (n_classes, n_features + 1), optional
            Model parameters. If None, uses self.theta_
            
        Returns:
        --------
        loss : float
            Negative log-likelihood with L2 regularization.
        """
        if theta is None:
            theta = self.theta_
            
        n_samples = X.shape[0]
        z = np.dot(X, theta.T)
        y_proba = self._softmax(z)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-15
        y_proba = np.clip(y_proba, epsilon, 1.0 - epsilon)
        
        # Cross-entropy loss
        log_likelihood = np.sum(y * np.log(y_proba)) / n_samples
        
        # L2 regularization
        theta_clipped = np.clip(theta[:, 1:], -100, 100)
        reg_term = 0.5 / max(self.C, 1e-10) * np.sum(theta_clipped ** 2)
        
        # Return negative log-likelihood + regularization
        return -log_likelihood + reg_term
        
    def _compute_gradient(self, X, y, y_proba):
        """
        Compute gradient of the loss function
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_features + 1)
            Input data with intercept
            
        y : array, shape (n_samples, n_classes)
            One-hot encoded target values
            
        y_proba : array, shape (n_samples, n_classes)
            Predicted probabilities
            
        Returns:
        --------
        gradient : array, shape (n_classes, n_features + 1)
            Gradient of the loss function
        """
        n_samples = X.shape[0]
        error = y_proba - y
        
        # Gradient of log-likelihood
        gradient = np.dot(error.T, X) / n_samples
        
        # Add gradient of L2 regularization term
        gradient_reg = np.zeros_like(gradient)
        theta_clipped = np.clip(self.theta_[:, 1:], -100, 100)
        gradient_reg[:, 1:] = theta_clipped / max(self.C, 1e-10)
        
        return gradient + gradient_reg
    
    def fit(self, X, y):
        """
        Fit the model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((len(y), self.n_classes_))
        for i, cls in enumerate(y):
            y_one_hot[i, self.class_to_idx_[cls]] = 1
        
        # Add intercept term to X
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape
        self.n_features_ = n_features - 1
        
        # Initialize parameters
        self.theta_ = self._initialize_params(self.n_features_, self.n_classes_)
        self.coef_ = self.theta_[:, 1:]
        self.intercept_ = self.theta_[:, 0]
        
        # Store loss history
        self.loss_history_ = []
        
        # mini-batch gradient descent
        n_batches = 10
        batch_size = max(n_samples // n_batches, 1)
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y_one_hot[indices]
            
            # Process mini-batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                z_batch = np.dot(X_batch, self.theta_.T)
                y_proba_batch = self._softmax(z_batch)
                
                # Compute gradient
                gradient = self._compute_gradient(X_batch, y_batch, y_proba_batch)
                
                # Clip gradients to prevent exploding gradients
                gradient = np.clip(gradient, -10, 10)
                
                # Adaptive learning rate with decay
                lr = self.learning_rate / (1 + 0.001 * iteration)
                
                # Update parameters
                self.theta_ -= lr * gradient
            
            # Compute loss for monitoring
            try:
                loss = self._compute_loss(X_with_intercept, y_one_hot)
                self.loss_history_.append(loss)
                self.coef_ = self.theta_[:, 1:]
                self.intercept_ = self.theta_[:, 0]
                
                # Check for convergence
                loss_change = prev_loss - loss
                if abs(loss_change) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
                    
                prev_loss = loss
            except Exception as e:
                print(f"Warning: {e} at iteration {iteration}")
                continue
            
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples
            
        Returns:
        --------
        P : array, shape (n_samples, n_classes)
            The class probabilities of the input samples
        """
        X = np.asarray(X)
        X_with_intercept = self._add_intercept(X)
        z = np.dot(X_with_intercept, self.theta_.T)
        return self._softmax(z)
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples
            
        Returns:
        --------
        y : array, shape (n_samples,)
            The predicted classes
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return np.array([self.classes_[idx] for idx in indices])


class Pipeline:
    def __init__(self, count_vec, tfidf_transformer, scaler, clf):
        self.count_vec = count_vec
        self.tfidf_transformer = tfidf_transformer
        self.scaler = scaler
        self.clf = clf
        self.classes_ = clf.classes_
    
    def predict(self, X):
        """Predict class labels for X."""
        X_counts = self.count_vec.transform(X)
        X_tfidf = self.tfidf_transformer.transform(X_counts)
        X_scaled = self.scaler.transform(X_tfidf)
        return self.clf.predict(X_scaled.toarray())
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        X_counts = self.count_vec.transform(X)
        X_tfidf = self.tfidf_transformer.transform(X_counts)
        X_scaled = self.scaler.transform(X_tfidf)
        return self.clf.predict_proba(X_scaled.toarray())


def test_model_on_examples(model, examples, preprocess_text):
    """Test the trained model on example sentences."""
    
    # Preprocess the examples
    processed_examples = []
    for example in examples:
        _, processed = preprocess_text(example)
        processed_examples.append(processed)
    
    # Make predictions
    predictions = model.predict(processed_examples)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(processed_examples)
    
    # Print results
    print("\nModel predictions on example sentences:")
    print("{:<5} {:<15} {:<15} {:<60}".format("No.", "Prediction", "Confidence", "Sentence"))
    print("-" * 95)
    
    for i, (example, prediction, proba) in enumerate(zip(examples, predictions, probabilities), 1):
        # Get confidence score
        class_idx = list(model.classes_).index(prediction)
        confidence = proba[class_idx]
        
        # Truncate long sentences
        display_text = example if len(example) < 60 else example[:57] + "..."
        print("{:<5d} {:<15} {:<15.4f} {:<60}".format(i, prediction, confidence, display_text))

def build_and_evaluate_model(df, test_size=0.2, random_state=42):
    """
    Build and evaluate model with CV = 5
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import numpy as np
    X = df['processed_text']
    y = df['sentiment']
    print(f"\nClass distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Create feature extraction pipeline
    count_vec = CountVectorizer(min_df=3)
    X_counts = count_vec.fit_transform(X)
    
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    X_tfidf_array = X_tfidf.toarray()
    
    # Normalize features
    scaler = StandardScaler(with_mean=False)
    X_tfidf_scaled = scaler.fit_transform(X_tfidf)
    X_tfidf_array = X_tfidf_scaled.toarray()
    
    # Test regularization values
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print("\nPerforming 5-fold cross-validation with different regularization levels")
    
    cv_results = {}
    
    for C in C_values:
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X_tfidf_array):
            # Split data
            X_train_fold, X_val_fold = X_tfidf_array[train_idx], X_tfidf_array[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx].values, y.iloc[val_idx].values
            
            # Create and train model
            model = MultinomialLogisticRegression(
                C=C,
                max_iter=1000,
                learning_rate=0.1,
                random_state=random_state,
                verbose=False
            )
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = model.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(accuracy)
        
        # Calculate mean accuracy across folds
        mean_accuracy = np.mean(fold_scores)
        cv_results[C] = mean_accuracy
        
        print("{:<8.2f} {:<15.4f}".format(C, mean_accuracy))
    
    # Find best C value
    best_C = max(cv_results, key=cv_results.get)
    print(f"\nBest inverse regularization parameter (C): {best_C}")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Process training data
    X_train_counts = count_vec.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
    
    # Train final model with best parameters
    final_model = MultinomialLogisticRegression(
        C=best_C,
        max_iter=1000,
        learning_rate=0.1,
        random_state=random_state,
        verbose=True
    )
    
    # Train the model
    final_model.fit(X_train_tfidf_scaled.toarray(), y_train.values)
    pipeline = Pipeline(count_vec, tfidf_transformer, scaler, final_model)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Accuracy on Test Set: {overall_accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot loss history
    if len(final_model.loss_history_) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(final_model.loss_history_)
        plt.title('Loss History During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_history.png')
    
    # Plot CV results
    plt.figure(figsize=(10, 6))
    plt.plot(list(cv_results.keys()), list(cv_results.values()), 'o-')
    plt.xscale('log')
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Cross-Validation Performance vs. Regularization')
    plt.grid(True)
    plt.savefig('cv_results.png')
    
    return pipeline, overall_accuracy