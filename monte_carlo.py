# %%
import matplotlib.pyplot as plt
import pyro
import seaborn as sns
import torch
from pyro.infer.mcmc import HMC, MCMC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# depends on your network connection,
# it make take more than a few mintues
mnist = fetch_openml("mnist_784")

x_train, x_test, y_train, y_test = train_test_split(
    mnist.data,
    mnist.target,
    test_size=0.2,
    random_state=2020,
)


# %%
# create a DecisionTreeClassifier object and fit with training data
# clf_dt = ...
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml("mnist_784")

X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# %%
# create a LogisticRegression object and fit with training data
# clf_lr = ...
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = fetch_openml("mnist_784")

# Split the data into features (X) and labels (y)
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a LogisticRegression object
clf_lr = LogisticRegression(random_state=42, max_iter=100)  # You can adjust max_iter based on your needs
# Fit the logistic regression model with training data
clf_lr.fit(X_train, y_train)
# Make predictions on the test set
y_pred_lr = clf_lr.predict(X_test)
# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")


# %%
# get prediction label using clf_dt and print the accuracy score
# y_pred_dt = ...
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml("mnist_784")
# Split the data into features (X) and labels (y)
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming you have a trained DecisionTreeClassifier named clf_dt
# If not, you should train it before using it for predictions
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
# Get predictions on the test set
y_pred_dt = clf_dt.predict(X_test)
# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")



# %%
# get prediction label using clf_lr and print the accuracy score
# y_pred_lr = ...
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml("mnist_784")
# Split the data into features (X) and labels (y)
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming you have a trained LogisticRegression named clf_lr
# If not, you should train it before using it for predictions
clf_lr = LogisticRegression(random_state=42, max_iter=100)  # You can adjust max_iter based on your needs
clf_lr.fit(X_train, y_train)
# Get predictions on the test set
y_pred_lr = clf_lr.predict(X_test)
# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")


# %%
# label to be used in Monte Carlo method
y_dt = torch.Tensor(y_pred_dt == y_test)
y_lr = torch.Tensor(y_pred_lr == y_test)


# %%
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC
def model():
    # create a variable underlying_p for the Bernoulli distribution
    # where underlying_p is a sample of Uniform distribution from 0 to 1
    # underlying_p = ...
    underlying_p = pyro.sample("underlying_p", dist.Uniform(0, 1))
    # create a hidden Bernoulli distribution with p = underlying_p
    # y_hidden_dist = ...
    y_hidden_dist = dist.Bernoulli(probs=underlying_p)
    # sample the label from the hidden Bernoulli distribution
    y_real = pyro.sample("obs", y_hidden_dist)
    return y_real


def conditioned_model(model, y):
    conditioned_model_function = pyro.poutine.condition(
        model, data={"obs": y}
    )
    return conditioned_model_function()


def monte_carlo(y):
    pyro.clear_param_store()
    # create a Simple Hamiltonian Monte Carlo kernel with step_size of 0.1
    # hmc_kernel = ...
    hmc_kernel = HMC(model, step_size=0.1, trajectory_length=1)
    # create a Markov Chain Monte Carlo method with:
    # the hmc_kernel, 500 samples, and 100 warmup iterations
    # mcmc = ...
    mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)

    mcmc.run(model, y)

    sample_dict = mcmc.get_samples(num_samples=5000)
    plt.figure(figsize=(8, 6))
    sns.distplot(sample_dict["p"].numpy())
    plt.xlabel("Observed probability value")
    plt.ylabel("Observed frequency")
    plt.show()
    mcmc.summary(prob=0.95)

    return sample_dict


# %%
# run the Monte Carlo method with y_dt and save the sample_dict with name simulations_dt
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import torch
# Load the MNIST dataset for illustration
mnist = fetch_openml("mnist_784")
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming you have a DecisionTreeClassifier named clf
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Convert the decision tree parameters to a PyTorch tensor
decision_tree_params = torch.tensor([tree.tree_ for tree in clf.estimators_], requires_grad=True)

def model(X, y=None):
    # Priors for decision tree parameters
    tree_params_prior = dist.Normal(0, 1).expand([len(clf.estimators_), -1]).to_event(1)
    
    # Use a DecisionTreeClassifier with Bayesian parameters
    clf_bayesian = DecisionTreeClassifier(random_state=42)
    clf_bayesian.estimators_ = [tree.tree_ for tree in decision_tree_params]

    # Likelihood
    y_pred = clf_bayesian.predict(X)
    with pyro.plate("data", size=len(y)):
        obs = pyro.sample("obs", dist.Categorical(logits=y_pred), obs=y)

# Perform MCMC inference
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100)
mcmc.run(X_train, y_train)

# Get the posterior samples
posterior_samples = mcmc.get_samples()

# Save the sample_dict with the name simulations_dt
torch.save(posterior_samples, "simulations_dt.pt")

#%%
# run the Monte Carlo method with y_lr and save the sample_dict with name simulations_lr
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import torch
# Load the MNIST dataset for illustration
mnist = fetch_openml("mnist_784")
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming you have a DecisionTreeClassifier named clf
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Convert the decision tree parameters to a PyTorch tensor
decision_tree_params = torch.tensor([tree.tree_ for tree in clf.estimators_], requires_grad=True)

def model(X, y=None):
    # Priors for decision tree parameters
    tree_params_prior = dist.Normal(0, 1).expand([len(clf.estimators_), -1]).to_event(1)



# %%
plt.figure(figsize=(8, 6))
sns.distplot(
    simulations_dt["p"].numpy(),
    label="DecisionTree",
    color="red",
)
sns.distplot(
    simulations_lr["p"].numpy(),
    label="LogisticRegression",
    color="green",
)
plt.legend()
plt.show()
