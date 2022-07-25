

  
def estimator_lr():
  from sklearn.linear_model import LinearRegression
  return LinearRegression()


def estimator_sgd():
  from sklearn.linear_model import SGDRegressor
  return SGDRegressor(random_state=42)
  
  

