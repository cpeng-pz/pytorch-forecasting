# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # How to use custom data and implement custom models and metrics
# %% [markdown]
# ## Building a simple, first model
# %% [markdown]
# For demonstration purposes we will choose a simple fully connected model. It takes a timeseries of size `input_size` as input and outputs a new timeseries of size `output_size`. You can think of this `input_size` encoding steps and `output_size` decoding/prediction steps.

# %%
import os
import warnings

# warnings.filterwarnings("ignore")

os.chdir("../../..")


# %%
import torch
from torch import nn


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x)


# test that network works as intended
network = FullyConnectedModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
x = torch.rand(20, 5)
network(x).shape


# %%
from typing import Dict

from pytorch_forecasting.models import BaseModel


class FullyConnectedModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input)

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])


model = FullyConnectedModel(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)

# %% [markdown]
# This is a very basic implementation that could be readily used for training. But before we add additional features, let's first have a look how we pass data to this model.
# %% [markdown]
# ### Passing data to a model

# %%
import numpy as np
import pandas as pd

test_data = pd.DataFrame(
    dict(
        value=np.random.rand(30) - 0.5,
        #value=np.arange(30),
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
    )
)
test_data


# %%
from pytorch_forecasting import TimeSeriesDataSet

# create the dataset from the pandas dataframe
dataset = TimeSeriesDataSet(
    test_data,
    group_ids=["group"],
    target="value",
    time_idx="time_idx",
    min_encoder_length=5,
    max_encoder_length=5,
    min_prediction_length=2,
    max_prediction_length=2,
    time_varying_unknown_reals=["value"],
)


# %%
dataset.get_parameters()

# %% [markdown]
# Now, we take a look at the output of the dataloader. It's `x` will be fed to the model's forward method, that is why it is so important to understand it.

# %%
# convert the dataset to a dataloader
dataloader = dataset.to_dataloader(batch_size=4)

# and load the first batch
x, y = next(iter(dataloader))
print("x =", x)
print("\ny =", y)
print("\nsizes of x =")
for key, value in x.items():
    print(f"\t{key} = {value.size()}")

# %% [markdown]
# This explains why we had to first extract the correct input in our simple `FullyConnectedModel` above before passing it to our `FullyConnectedModule`.
# As a reminder:
#        

# %%
def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # x is a batch generated based on the TimeSeriesDataset
    network_input = x["encoder_cont"].squeeze(-1)
    prediction = self.network(network_input)

    # We need to return a dictionary that at least contains the prediction and the target_scale.
    # The parameter can be directly forwarded from the input.
    return dict(prediction=prediction, target_scale=x["target_scale"])

# %% [markdown]
# For such a simple architecture, we can ignore most of the inputs in ``x``. You do not have to worry about moving tensors to specifc GPUs, [PyTorch Lightning](https://pytorch-lightning.readthedocs.io) will take care of this for you.
# 
# Now, let's check if our model works:

# %%
x, y = next(iter(dataloader))
model(x)


# %%
dataset.x_to_index(x)

# %% [markdown]
# ### Coupling datasets and models

# %%
class FullyConnectedModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input).unsqueeze(-1)

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals) == 1
            and dataset.time_varying_unknown_reals[0] == dataset.target
        ), "Only covariate should be the target in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)

# %% [markdown]
# Now, let's initialize from our dataset:

# %%
model = FullyConnectedModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2)
model.summarize("full")  # print model summary
model.hparams

# %% [markdown]
# ### Defining additional hyperparameters

# %%
model.hparams


# %%
print(BaseModel.__init__.__doc__)

# %% [markdown]
# ## Classification

# %%
classification_test_data = pd.DataFrame(
    dict(
        target=np.random.choice(["A", "B", "C"], size=30),  # CHANGING values to predict to a categorical
        value=np.random.rand(30),  # INPUT values - see next section on covariates how to use categorical inputs
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
    )
)
classification_test_data


# %%
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# create the dataset from the pandas dataframe
classification_dataset = TimeSeriesDataSet(
    classification_test_data,
    group_ids=["group"],
    target="target",  # SWITCHING to categorical target
    time_idx="time_idx",
    min_encoder_length=5,
    max_encoder_length=5,
    min_prediction_length=2,
    max_prediction_length=2,
    time_varying_unknown_reals=["value"],
    target_normalizer=NaNLabelEncoder(),  # Use the NaNLabelEncoder to encode categorical target
)

x, y = next(iter(classification_dataset.to_dataloader(batch_size=4)))
y[0]  # target values are encoded categories

# The keyword argument ``target_normalizer`` is here redundant because the would have detected that a categorical target is used and therefore a :py:class:`~pytorch_forecasting.data.encoders.NaNLabelEncoder` is required.
# %%
from pytorch_forecasting.metrics import CrossEntropy


class FullyConnectedClassificationModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        n_classes: int,
        loss=CrossEntropy(),
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size * self.hparams.n_classes,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_cont"].size(0)
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input)
        # RESHAPE output to batch_size x n_decoder_timesteps x n_classes
        prediction = prediction.unsqueeze(-1).view(batch_size, -1, self.hparams.n_classes)

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        assert isinstance(dataset.target_normalizer, NaNLabelEncoder), "target normalizer has to encode categories"
        new_kwargs = {
            "n_classes": len(
                dataset.target_normalizer.classes_
            ),  # ADD number of classes as encoded by the target normalizer
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals) == 1
        ), "Only covariate should be in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)


model = FullyConnectedClassificationModel.from_dataset(classification_dataset, hidden_size=10, n_hidden_layers=2)
model.summarize("full")
model.hparams


# %%
# passing x through model
model(x)["prediction"].shape

# %% [markdown]
# ## Predicting multiple targets at the same time
# %% [markdown]
# Training a model to predict multiple targets simulateneously is not difficult to implement. We can even employ mixed targets, i.e. a mix of categorical and continous targets. The first step is to use define a dataframe with multiple targets:

# %%
multi_target_test_data = pd.DataFrame(
    dict(
        target1=np.random.rand(30),
        target2=np.random.rand(30),
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
    )
)
multi_target_test_data


# %%
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer

# create the dataset from the pandas dataframe
multi_target_dataset = TimeSeriesDataSet(
    multi_target_test_data,
    group_ids=["group"],
    target=["target1", "target2"],  # USING two targets
    time_idx="time_idx",
    min_encoder_length=5,
    max_encoder_length=5,
    min_prediction_length=2,
    max_prediction_length=2,
    time_varying_unknown_reals=["target1", "target2"],
    target_normalizer=MultiNormalizer(
        [EncoderNormalizer(), TorchNormalizer()]
    ),  # Use the NaNLabelEncoder to encode categorical target
)

x, y = next(iter(multi_target_dataset.to_dataloader(batch_size=4)))
y[0]  # target values are a list of targets


# %%
from typing import List, Union

from pytorch_forecasting.metrics import MAE, SMAPE, MultiLoss
from pytorch_forecasting.utils import to_list


class FullyConnectedMultiTargetModel(BaseModel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        target_sizes: Union[int, List[int]] = [],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size * len(to_list(self.hparams.target_sizes)),
            output_size=self.hparams.output_size * sum(to_list(self.hparams.target_sizes)),
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_cont"].size(0)
        network_input = x["encoder_cont"].view(batch_size, -1)
        prediction = self.network(network_input)
        # RESHAPE output to batch_size x n_decoder_timesteps x sum_of_target_sizes
        prediction = prediction.unsqueeze(-1).view(batch_size, self.hparams.output_size, sum(self.hparams.target_sizes))
        # RESHAPE into list of batch_size x n_decoder_timesteps x target_sizes[i] where i=1..len(target_sizes)
        stops = np.cumsum(self.hparams.target_sizes)
        starts = stops - self.hparams.target_sizes
        prediction = [prediction[..., start:stop] for start, stop in zip(starts, stops)]
        if isinstance(self.hparams.target_sizes, int):  # only one target
            prediction = prediction[0]

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        # By default only handle targets of size one here, categorical targets would be of larger size
        new_kwargs = {
            "target_sizes": [1] * len(to_list(dataset.target)),
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals)
            == len(dataset.target_names)  # Expect as as many unknown reals as targets
        ), "Only covariate should be in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)


model = FullyConnectedMultiTargetModel.from_dataset(
    multi_target_dataset,
    hidden_size=10,
    n_hidden_layers=2,
    loss=MultiLoss(metrics=[MAE(), SMAPE()], weights=[2.0, 1.0]),
)
model.summarize("full")
model.hparams

# %% [markdown]
# Now, let's pass some data through our model and calculate the loss.

# %%
out = model(x)
out


# %%
y_hat = model.transform_output(
    out
)  # the model's transform_output method re-scales/de-normalizes the predictions to into the real target space
model.loss(y_hat, y)

# %% [markdown]
# ## Using covariates

# %%
from pytorch_forecasting.models.base_model import BaseModelWithCovariates

print(BaseModelWithCovariates.__doc__)


# %%
from typing import Dict, List, Tuple

from pytorch_forecasting.models.nn import MultiEmbedding


class FullyConnectedModelWithCovariates(BaseModelWithCovariates):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(
            embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)

        # create network that will be fed with continious variables and embeddings
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size * n_features,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_lengths"].size(0)
        embeddings = self.input_embeddings(x["encoder_cat"])  # returns dictionary with embedding tensors
        network_input = torch.cat(
            [x["encoder_cont"]]
            + [
                emb
                for name, emb in embeddings.items()
                if name in self.encoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        prediction = self.network(network_input.view(batch_size, -1))

        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)

# %% [markdown]
# Note that the model does not make use of the known covariates in the decoder - this is obviously suboptimal but not scope of this tutorial. Anyways, let us create a new dataset with categorical variables and see how the model can be instantiated from it.

# %%
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

test_data_with_covariates = pd.DataFrame(
    dict(
        # as before
        value=np.random.rand(30),
        group=np.repeat(np.arange(3), 10),
        time_idx=np.tile(np.arange(10), 3),
        # now adding covariates
        categorical_covariate=np.random.choice(["a", "b"], size=30),
        real_covariate=np.random.rand(30),
    )
).astype(
    dict(group=str)
)  # categorical covariates have to be of string type
test_data_with_covariates


# %%
# create the dataset from the pandas dataframe
dataset_with_covariates = TimeSeriesDataSet(
    test_data_with_covariates,
    group_ids=["group"],
    target="value",
    time_idx="time_idx",
    min_encoder_length=5,
    max_encoder_length=5,
    min_prediction_length=2,
    max_prediction_length=2,
    time_varying_unknown_reals=["value"],
    time_varying_known_reals=["real_covariate"],
    time_varying_known_categoricals=["categorical_covariate"],
    static_categoricals=["group"],
)

model = FullyConnectedModelWithCovariates.from_dataset(dataset_with_covariates, hidden_size=10, n_hidden_layers=2)
model.summarize("full")  # print model summary
model.hparams

# %% [markdown]
# To test that the model could be trained, pass a sample batch.

# %%
x, y = next(iter(dataset_with_covariates.to_dataloader(batch_size=4)))  # generate batch
model(x)  # pass batch through model

# %% [markdown]
# ## Implementing an autoregressive / recurrent model

# %%
from torch.nn.utils import rnn

from pytorch_forecasting.models.base_model import AutoRegressiveBaseModel
from pytorch_forecasting.models.nn import LSTM


class LSTMModel(AutoRegressiveBaseModel):
    def __init__(
        self,
        target: str,
        target_lags: Dict[str, Dict[str, int]],
        n_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        # arguments target and target_lags are required for autoregressive models
        # even though target_lags cannot be used without covariates
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # use version of LSTM that can handle zero-length sequences
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=1,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hparams.hidden_size, 1)

    def encode(self, x: Dict[str, torch.Tensor]):
        # we need at least one encoding step as because the target needs to be lagged by one time step
        # because we use the custom LSTM, we do not have to require encoder lengths of > 1
        # but can handle lengths of >= 1
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging

        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        _, hidden_state = self.lstm(
            input_vector, lengths=effective_encoder_lengths, enforce_sorted=False  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)

            # transform into right shape
            prediction = self.output_layer(lstm_output)

            # predictions are not yet rescaled
            return dict(prediction=prediction, target_scale=x["target_scale"])

        else:  # prediction mode
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                # overwrite at target positions
                x[:, 0, target_pos] = lagged_targets[-1]  # take most recent target (i.e. lag=1)
                lstm_output, hidden_state = self.lstm(x, hidden_state)
                # transform into right shape
                prediction = self.output_layer(lstm_output)[:, 0]  # take first timestep
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )

            # predictions are already rescaled
            return dict(prediction=output, output_transformation=None, target_scale=x["target_scale"])

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state
        return output


model = LSTMModel.from_dataset(dataset, n_layers=2, hidden_size=10)
model.summarize("full")
model.hparams


# %%
x, y = next(iter(dataloader))

print(
    "prediction shape in training:", model(x)["prediction"].size()
)  # batch_size x decoder time steps x 1 (1 for one target dimension)
model.eval()  # set model into eval mode to use autoregressive prediction
print("prediction shape in inference:", model(x)["prediction"].size())  # should be the same as in training

# %% [markdown]
# ## Using and defining a custom/non-trivial metric
# %% [markdown]
# To use a different metric, simply pass it to the model when initializing it (preferably via the `from_dataset()` method). For example, to use mean absolute error with our `FullyConnectedModel` from the beginning of this tutorial, type

# %%
from pytorch_forecasting.metrics import MAE

model = FullyConnectedModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2, loss=MAE())
model.hparams

# %% [markdown]
# Note that some metrics might require a certain form of model prediction, e.g. quantile prediction assumes an output of shape `batch_size x n_decoder_timesteps x n_quantiles` instead of `batch_size x n_decoder_timesteps`. For the `FullyConnectedModel`, this means that we need to use a modified `FullyConnectedModule`network. Here `n_outputs` corresponds to the number of quantiles.

# %%
import torch
from torch import nn


class FullyConnectedMultiOutputModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, n_outputs: int):
        super().__init__()

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        self.n_outputs = n_outputs
        module_list.append(
            nn.Linear(hidden_size, output_size * n_outputs)
        )  # <<<<<<<< modified: replaced output_size with output_size * n_outputs

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x).reshape(x.size(0), -1, self.n_outputs)  # <<<<<<<< modified: added reshape


# test that network works as intended
network = FullyConnectedMultiOutputModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2, n_outputs=7)
network(torch.rand(20, 5)).shape  # <<<<<<<<<< instead of shape (20, 2), returning additional dimension for quantiles

# %% [markdown]
# ### Implement a new metric

# %%
from pytorch_forecasting.metrics import MultiHorizonMetric


class MAE(MultiHorizonMetric):
    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss

# %% [markdown]
# ### Model ouptut cannot be readily converted to prediction

# %%
from copy import copy

from pytorch_forecasting.metrics import NormalDistributionLoss


class FullyConnectedForDistributionLossModel(BaseModel):  # we inherit the `from_dataset` method
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, **kwargs):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedMultiOutputModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
            n_outputs=2,  # <<<<<<<< we predict two outputs for mean and scale of the normal distribution
        )
        self.loss = NormalDistributionLoss()

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals) == 1
            and dataset.time_varying_unknown_reals[0] == dataset.target
        ), "Only covariate should be the target in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input)  # shape batch_size x n_decoder_steps x 2
        if (
            self.training or n_samples is None
        ):  # training is a PyTorch variable indicating if a module is being trained (tracing gradients) or evaluated
            assert n_samples is None, "We need to predict parameters when training"
            output_transformation = True
        else:
            # let's sample from our distribution - first we need to scale the parameters to real space
            scaled_parameters = self.transform_output(
                dict(
                    prediction=prediction,
                    target_scale=x["target_scale"],
                )
            )
            # and then sample from distribution
            prediction = self.loss.sample(scaled_parameters, n_samples)
            output_transformation = None  # predictions are already re-scaled
        return dict(prediction=prediction, target_scale=x["target_scale"], output_transformation=output_transformation)

    def transform_output(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        # this is already implemented in pytorch forecasting but this code demonstrates the point
        # input is forward's output
        # depending on output, transform differently
        if out.get("output_transformation", True) is None:  # samples are already rescaled
            out = out["prediction"]
        else:  # parameters need to be rescaled
            out = self.loss.rescale_parameters(
                out["prediction"], target_scale=out["target_scale"], encoder=self.output_transformer
            )
        return out


model = FullyConnectedForDistributionLossModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2)
model.summarize("full")
model.hparams


# %%
x["decoder_lengths"]


# %%
x, y = next(iter(dataloader))

print("parameter predition shape: ", model(x)["prediction"].size())
model.eval()  # set model into eval mode for sampling
print("sample prediction shape: ", model(x, n_samples=200)["prediction"].size())


# %%
model.predict(dataloader, mode="quantiles", n_samples=100).shape

# %% [markdown]
# The returned quantiles are here determined by the quantiles defined in the loss function and can be modified by passing a list of quantiles to at initialization.

# %%
model.loss.quantiles


# %%
NormalDistributionLoss(quantiles=[0.2, 0.8]).quantiles

# %% [markdown]
# ## Adding custom plotting and interpretation
# %% [markdown]
# ### Log often whenever an example prediction vs actuals plot is created

# %%
import matplotlib.pyplot as plt


def plot_prediction(
    self,
    x: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    idx: int,
    plot_attention: bool = True,
    add_loss_to_title: bool = False,
    show_future_observed: bool = True,
    ax=None,
) -> plt.Figure:
    """
    Plot actuals vs prediction and attention

    Args:
        x (Dict[str, torch.Tensor]): network input
        out (Dict[str, torch.Tensor]): network output
        idx (int): sample index
        plot_attention: if to plot attention on secondary axis
        add_loss_to_title: if to add loss to title. Default to False.
        show_future_observed: if to show actuals for future. Defaults to True.
        ax: matplotlib axes to plot on

    Returns:
        plt.Figure: matplotlib figure
    """
    # plot prediction as normal
    fig = super().plot_prediction(
        x, out, idx=idx, add_loss_to_title=add_loss_to_title, show_future_observed=show_future_observed, ax=ax
    )

    # add attention on secondary axis
    if plot_attention:
        interpretation = self.interpret_output(out)
        ax = fig.axes[0]
        ax2 = ax.twinx()
        ax2.set_ylabel("Attention")
        encoder_length = x["encoder_lengths"][idx]
        ax2.plot(
            torch.arange(-encoder_length, 0),
            interpretation["attention"][idx, :encoder_length].detach().cpu(),
            alpha=0.2,
            color="k",
        )
    fig.tight_layout()
    return fig

# %% [markdown]
# ### Log at the end of an epoch

# %%
def step(
    self, x: Dict[str, torch.Tensor], y: torch.Tensor, batch_idx: int, **kwargs
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Run for each train/val step.

    Args:
        x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
        y (torch.Tensor): y as passed to the loss function by the dataloader
        batch_idx (int): batch number
        **kwargs: additional arguments to pass to the network apart from ``x``

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: tuple where the first
            entry is a dictionary to which additional logging results can be added for consumption in the
            ``epoch_end`` hook and the second entry is the model's output.
    """
    # extract data and run model
    log, out = super().step(x, y, batch_idx)
    # calculate interpretations etc for latter logging
    if self.log_interval > 0:
        detached_output = {name: tensor.detach() for name, tensor in out.items()}
        interpretation = self.interpret_output(
            detached_output,
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        log["interpretation"] = interpretation
    return log, out


def epoch_end(self, outputs):
    """
    Run at epoch end for training or validation
    """
    if self.log_interval > 0:
        self.log_interpretation(outputs)

# %% [markdown]
# ### Log at the end of training

# %%
def on_fit_end(self):
    """
    run at the end of training
    """
    if self.log_interval > 0:
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.cpu(), metadata=labels, tag=name, global_step=self.global_step
            )

# %% [markdown]
# ## Minimal testing of models
# %% [markdown]
# Testing models is essential to quickly detect problems and iterate quickly. Some issues can be only identified after lengthy training but many problems show up after one or two batches. PyTorch Lightning, on which PyTorch Forecasting is built, makes it easy to set up such tests.

# %%
from pytorch_lightning import Trainer

model = FullyConnectedForDistributionLossModel.from_dataset(dataset, hidden_size=10, n_hidden_layers=2, log_interval=1)
trainer = Trainer(fast_dev_run=True)
trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)


