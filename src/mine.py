import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd

# Our neural network used by MINE
from models import mine_net


# Use GPU if available, otherwise CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class mine:

    def __init__(
        self,
        p_dis,
        q_dis,
        num_iterations,
        all=True,
        batch_size=512,
        lr=0.0001
    ):

        # Learning rate
        self.lr = lr

        self.all = all

        # Window used for moving average
        self.ma_window_size = 1

        # Batch size cannot be bigger than dataset
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size

        # Convert first input into tensor
        if not isinstance(p_dis, torch.Tensor):
            self.obs = torch.tensor(
                p_dis,
                dtype=torch.float32
            )
        else:
            self.obs = p_dis.float()

        # Convert second input into tensor
        if not isinstance(q_dis, torch.Tensor):
            self.acs = torch.tensor(
                q_dis,
                dtype=torch.float32
            )
        else:
            self.acs = q_dis.float()

        # Make obs 2D if it is 1D
        if self.obs.dim() == 1:
            self.obs = self.obs.unsqueeze(1)

        # Make acs 2D if it is 1D
        if self.acs.dim() == 1:
            self.acs = self.acs.unsqueeze(1)

        # Number of training iterations
        self.num_iterations = num_iterations

        # Run experiment 3 times
        self.expts = 3


    def kullback_liebler(self, dis_p, dis_q, kl_net):

        # Score for real X,Y pairs
        t = kl_net(dis_p)

        # Score for shuffled X,Y pairs
        et = torch.exp(
            kl_net(dis_q)
        )

        # Mutual Information lower bound
        mi_lb = (
            torch.mean(t)
            - torch.log(torch.mean(et))
        )

        return mi_lb, t, et


    def learn_klne(
        self,
        batch,
        mine_net,
        mine_net_optim,
        ma_et,
        ma_rate=0.001
    ):

        # Get real and shuffled batches
        joint, marginal = batch

        # Convert joint batch to tensor and send to device
        joint = torch.FloatTensor(joint).to(device)

        # Convert marginal batch to tensor and send to device
        marginal = torch.FloatTensor(marginal).to(device)

        # Calculate MI
        mi_lb, t, et = self.kullback_liebler(
            joint,
            marginal,
            mine_net
        )

        # Update moving average
        ma_et = (
            (1 - ma_rate) * ma_et
            + ma_rate * torch.mean(et)
        )

        # Loss used to train MINE
        loss = -(
            torch.mean(t)
            - (1 / ma_et.mean()).detach()
            * torch.mean(et)
        )

        # Remove old gradients
        mine_net_optim.zero_grad()

        # Calculate new gradients
        autograd.backward(loss)

        # Update network weights
        mine_net_optim.step()

        return mi_lb, ma_et


    def trip_sample_batch(
        self,
        sample_mode="joint"
    ):

        # Pick random rows
        index = np.random.choice(
            range(self.obs.shape[0]),
            size=self.batch_size,
            replace=False
        )

        # Create shuffled X,Y pairs
        if sample_mode == "marginal":

            # Pick different random rows for Y
            marginal_index = np.random.choice(
                range(self.obs.shape[0]),
                size=self.batch_size,
                replace=False
            )

            batch = np.concatenate(
                (
                    self.obs[index, :],
                    np.array(
                        self.acs[marginal_index, :]
                    )
                ),
                axis=1
            )

        # Create real X,Y pairs
        else:

            batch = np.concatenate(
                (
                    self.obs[index, :],
                    np.array(
                        self.acs[index, :]
                    )
                ),
                axis=1
            )

        return batch


    def trip_train(
        self,
        tripmine_net,
        tripmine_net_optim
    ):

        # Starting moving average
        ma_et = 1.0

        # Store MI values
        result = []

        # Train many times
        for i in range(self.num_iterations):

            # Create real batch
            # and shuffled batch
            batch = (
                self.trip_sample_batch(),
                self.trip_sample_batch(
                    sample_mode="marginal"
                )
            )

            # Train MINE for one step
            mi_lb, ma_et = self.learn_klne(
                batch,
                tripmine_net,
                tripmine_net_optim,
                ma_et
            )

            # Save MI value
            result.append(
                mi_lb.detach().cpu().numpy()
            )

        return result


    def ma(self, a):

        # Moving average of MI values
        return [
            np.mean(
                a[i:i + self.ma_window_size]
            )
            for i in range(
                0,
                len(a) - self.ma_window_size
            )
        ]


    def trip_initialiser(self):

        # Total input size = X columns + Y columns
        input_size = (
            self.obs.shape[1]
            + self.acs.shape[1]
        )

        # Create MINE neural network
        tripmine_net = mine_net(
            input_size
        ).to(device)

        # Create Adam optimizer
        tripmine_net_optim = optim.Adam(
            tripmine_net.parameters(),
            lr=self.lr
        )

        # Store results of experiments
        trip_results = []

        # Run experiment 3 times
        for expt in range(self.expts):

            result = self.trip_train(
                tripmine_net,
                tripmine_net_optim
            )

            # Apply moving average
            result = self.ma(result)

            trip_results.append(result)

        return np.array(trip_results)


    def run(self):

        # Start the complete MINE process
        results = self.trip_initialiser()

        return results