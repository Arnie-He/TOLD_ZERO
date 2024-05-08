import matplotlib.pyplot as plt
import pandas as pd

# plots
fig, ax = plt.subplots()
# Load the data
df1 = pd.read_csv("data/data.csv")  
# Extract the step and episode_reward columns
steps = df1['Step'].values
loss = df1['Loss'].values
returns_right = df1['Return right'].values
returns_left = df1['Return left'].values
return_right_stds = df1['Return right Std'].values
return_left_stds = df1['Return left Std'].values
loss_stds = df1['Loss Std'].values


# calculate the upper and lower bound of the returns and loss
up1 = [returns_right[i] + return_right_stds[i] for i in range(len(returns_right))]
down1 = [returns_right[i] - return_right_stds[i] for i in range(len(returns_right))]
up2 = [loss[i] + loss_stds[i] for i in range(len(loss))]
down2 = [loss[i] - loss_stds[i] for i in range(len(loss))]
up3 = [returns_left[i] + return_left_stds[i] for i in range(len(returns_right))]
down3 = [returns_left[i] - return_left_stds[i] for i in range(len(returns_right))]
# plot the data
ax.plot(steps, returns_right, label='TOLD-ZERO MCTS, Cartpole_v1, returns right', color='blue')
ax.plot(steps, returns_left, label='TOLD-ZERO MCTS, Cartpole_v1, returns left', color='green')
ax.plot(steps, loss, label='TOLD-ZERO MCTS, Cartpole_v1, loss', color='red')
# plot the upper and lower bound
ax.fill_between(steps, down1, up1, alpha=0.2)
ax.fill_between(steps, down2, up2, alpha=0.2)
ax.fill_between(steps, down3, up3, alpha=0.2)
# set the title and labels
ax.legend()
ax.set_xlabel('Steps')
ax.set_ylabel('Loss & Returns')
plt.grid(True, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig('outputs/cartpole_returns.png') 