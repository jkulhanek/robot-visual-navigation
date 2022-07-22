import unittest
import numpy as np
import torch
from .util import autocrop_observations, pixel_control_reward, pixel_control_loss, value_loss, reward_prediction_loss


class UtilTest(unittest.TestCase):
    def testAutocropShape(self):
        obs = torch.rand((10, 5, 3, 83, 83), dtype=torch.float32)
        cropped = autocrop_observations(obs, 4)
        self.assertTupleEqual(cropped.size(), (10, 5, 3, 80, 80))

    def testAutocropNoCropping(self):
        obs = torch.rand((10, 5, 3, 84, 84), dtype=torch.float32)
        cropped = autocrop_observations(obs, 4)
        self.assertTupleEqual(cropped.size(), (10, 5, 3, 84, 84))

    def testAutocropOutputShapeSupplied(self):
        obs = torch.rand((10, 5, 3, 84, 84), dtype=torch.float32)
        cropped = autocrop_observations(obs, 4, (10, 10))
        self.assertTupleEqual(cropped.size(), (10, 5, 3, 40, 40))

    def testPixelControlRewards(self):
        obs = torch.rand((10, 5, 3, 84, 84), dtype=torch.float32)
        rewards = pixel_control_reward(obs, cell_size=4, output_size=(20, 20))

        self.assertTupleEqual(rewards.size(), (10, 4, 1, 20, 20))

    def testPixelControlRewardValues(self):
        gamma = 0.9

        def _calc_pixel_change(state, last_state):
            d = np.absolute(state[2:-2, 2:-2, :] - last_state[2:-2, 2:-2, :])
            # (80,80,3)
            m = np.mean(d, 2)
            s = m.shape
            average_width = 4
            sh = s[0]//average_width, average_width, s[1]//average_width, average_width
            c = m.reshape(sh).mean(-1).mean(1)
            return c

        frames = [np.random.uniform(size=(84, 84, 3)) for _ in range(6)]
        pc_R = start = np.zeros([20, 20], dtype=np.float32)
        frames.reverse()

        results = []
        for i, frame in enumerate(frames[1:]):
            pixel_change = _calc_pixel_change(frames[i], frame)
            pc_R = pixel_change + gamma * pc_R
            results.append(pc_R)

        results.reverse()
        frames.reverse()

        # Compute results
        observations = np.stack(frames).astype(np.float32)
        observations = torch.from_numpy(observations)
        observations = observations.permute(0, 3, 1, 2).unsqueeze(0)
        pseudo_rewards = pixel_control_reward(
            observations, output_size=(20, 20))
        last_rewards = torch.from_numpy(
            start.astype(np.float32)).unsqueeze(0).unsqueeze(1)
        T = observations.size()[1] - 1
        for i in reversed(range(T)):
            previous_rewards = last_rewards if i + \
                1 == T else pseudo_rewards[:, i + 1]
            pseudo_rewards[:, i].add_(gamma, previous_rewards)

        pseudo_rewards = pseudo_rewards.view(5, 20, 20)
        self.assertTrue(np.allclose(pseudo_rewards.numpy(), np.stack(results)))

    def testPixelControlLoss(self):
        torch.manual_seed(1)
        obs = torch.rand((10, 5, 3, 84, 84), dtype=torch.float32)
        actions = torch.randint(0, 4, (10, 4))
        action_predictions = torch.rand(
            (10, 5, 4, 20, 20), dtype=torch.float32)

        loss = pixel_control_loss(obs, actions, action_predictions)

    def testValueLoss(self):
        torch.manual_seed(1)
        values = torch.rand((10, 5,), dtype=torch.float32)
        rewards = torch.rand((10, 4,), dtype=torch.float32)

        loss = value_loss(values, rewards, 0.99)

    def testRewardPredictionLoss(self):
        torch.manual_seed(1)
        predictions = torch.rand((10, 3,), dtype=torch.float32)
        rewards = torch.rand((10,), dtype=torch.float32)

        loss = reward_prediction_loss(predictions, rewards)


if __name__ == '__main__':
    unittest.main()
