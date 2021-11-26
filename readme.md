# Introduction to Machine Learning CW2, Part 2: Regression Model 

For project details creating and training a regression model on the California House Prices Dataset [[1]](https://econpapers.repec.org/article/eeestapro/v_3a33_3ay_3a1997_3ai_3a3_3ap_3a291-297.htm). The dataset contains 20,640 observations from the 1990 Census and details 10 distinct features with the median house value being the target variable. The 10 features are listed below:

* **longitude**: longitude of the block group
* **latitude**: latitude of the block group
* **housing median age**: median age of the individuals living in the block group
* ****total rooms****: total number of rooms in the block group
* ****total bedrooms****: total number of bedrooms in the block group
* **population**: total population of the block group
* **households**: number of households in the block group
* **median income**: median income of the households comprise in the block group
* **ocean proximity**: proximity to the ocean of the block group
* **median house value**: median value of the houses of the block group


## **Regressor class** 
- Inheritance: `torch.nn.Module`

| Attributes | Description |
| ----------- | ----------- |
| `input_size` | The number of columns (features) present in the training data. |
| `output_size` | 1 (predicting on single target variable) |
| `nb_epoch` | Number of epochs to attempt training model over. |
| `_sequential` | Calls `nn.Sequential()` to set layer types, layer sizes, and activation functions. |
| `y_scaler` | Contains `sklearn.preprocessing.MinMaxScaler()` for scaling target variable data. |
| `x_scaler` | As above, except for scaling all other features. |
| `history` | List of tuples, collected over each epoch in training process. This tuple stores epoch, loss at epoch, and validation loss at that epoch if validation dataset is passed to training function. |
| `hyper_params` | Contains hyperparameters of the model. In this case, arguments passed to the fit function. |


| Method | Description |
| ----------- | ----------- |
| `init` | Regressor constructor calls the super class constructor and sMets various class attributes. |
| `fit(X, y)` | Method to train a regression model using the training set (X, y). |
| `predict(X)` | Method to predict median house values from test set X. |
| `score(x, y)` | Method to evaluate the model accuracy on a validation dataset. |
| `_preprocessor(x, y)`| Preprocess network inputs by scaling and converting to tensors. |
| `_weight_regularization(loss, reg, decay)` | Performs either L1 or L2 weight regularization.|
| `_verbose()` | If called, print the epoch (if multiple of 10) and the loss and validation loss (if available) for that training step. |
| `plot_performance()` | If called, plot the training loss and validation loss (if available) curves over the epochs trained for. |
| `get_model()` | Return print out of the model shape and layer sizes. |

As well the parameters listed above, the `fit()` method has additional keyword arguments passed. These are detailed below:
* `lr` = `1e-3`
  * Learning rate.
* `opt_func` = `torch.optim.Adam`
  * Optimizer function: Adam.
* `batch_size` = `None`
  * For performing mini-batch gradient descent. Pass `None` to perform batch gradient descent.
* `regularizer` = `None`
  * Sets L1 or L2 regularization.
* `lambda_decay`= `1e-5`
  * Controls lambda value used in weight regularization.
* `callbacks` = `{}`
  * If an empty dictionary is passed than default values are passed within the `fit()` method to the variables controlled by callbacks. These variables include: `verbose` (whether training loss etc. is printed), `monitor` (whether to monitor loss or validation loss for early stopping), and `patience` (integer value dictating threshold for stopping training after so many epochs without improvement). Default values are respectively: `False`, `val_loss`, `10`.
* `validation_set` = `None`
  * To include a validation set, the proper notation would be [X_val, y_val], i.e. pass an iterable containing validation data and validation target variable.

Example usage of `Regressor`:
```
df = pd.read_csv('housing.csv', index_col=None)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split data into train, test, validation.
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

# Set callbacks.
callbacks = {
    "verbose" : True,
    "monitor" : "val_loss",
    "patience" : 8
}

# Create and train regression model.
regressor = Regressor(X_train, nb_epoch=500)
regressor.fit(X_train,y_train, callbacks=callbacks, validation_set=[X_val,y_val])

# Root mean squared error.
rmse = regressor.score(X_test, y_test)
print(rmse)

# Plot model performance.
regressor.plot_performance()
```

A gridsearch was undertaken to optimize hyperparameters of the regressor network. Partial results of this gridsearch are shown in the parallel coordinates plots below and a spreadsheet of the full results are located [here](https://github.com/LordLean/California-House-Prices/blob/main/Images/grid_search.csv).

Trimmed results (rmse<100000):
![](https://raw.githubusercontent.com/LordLean/California-House-Prices/main/Images/total.png)

Best performers, also tabulated below:
![](https://raw.githubusercontent.com/LordLean/California-House-Prices/main/Images/best.png)

| optimizer   |   learning_rate | regularizer   | regularizer_lambda   |   batch_size |    rmse |   epoch |   train_time (s) |
|:------------|----------------:|:--------------|:---------------------|-------------:|--------:|--------:|-------------:|
| RMSprop     |          0.0001 | None          | None                 |           64 | 52536   |     131 |     33.6131  |
| RMSprop     |          0.0005 | None          | None                 |           64 | 52672.4 |      57 |     14.881   |
| Adam        |          0.0001 | None          | None                 |           32 | 53262.3 |      53 |     28.3468  |
| RMSprop     |          0.001  | None          | None                 |           32 | 53328.9 |      43 |     20.3538  |
| Adam        |          0.001  | None          | None                 |          256 | 53621   |      71 |      7.84132 |
| Adam        |          0.001  | None          | None                 |          512 | 53672.9 |      40 |      3.39846 |
| Adam        |          0.0005 | l2            | 1e-05                |          128 | 53766.4 |      85 |     17.0314  |
| Adam        |          0.0005 | None          | None                 |          128 | 54168.5 |      39 |      6.69238 |
| Adam        |          0.001  | None          | None                 |           32 | 54197.7 |      38 |     20.4892  |
