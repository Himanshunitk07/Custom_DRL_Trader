import pandas as pd
import numpy as np
import os

def generate_csv_files(agent, env, training_results, test_results, output_dir="dashboard_data"):
    """
    Generates all necessary CSV files for the Streamlit dashboard.

    Args:
        agent (ActorCriticAgent): The trained agent.
        env (ShortTermDynamicTrader): The trading environment.
        training_results (tuple): A tuple containing (net_worths, episode_returns).
        test_results (list): A list of dictionaries from the test_agent function.
        output_dir (str): The directory to save the CSV files in.
    """
    print("\n" + "="*60)
    print("Generating CSV files for Streamlit Dashboard...")
    print("="*60)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 1. Generate Training Summary CSV
    try:
        net_worths, episode_returns = training_results
        df_training = pd.DataFrame({
            'episode': range(1, len(net_worths) + 1),
            'final_net_worth': net_worths,
            'portfolio_return_pct': episode_returns
        })
        training_filepath = os.path.join(output_dir, 'training_summary.csv')
        df_training.to_csv(training_filepath, index=False)
        print(f"Successfully saved training summary to: {training_filepath}")
    except Exception as e:
        print(f"Error saving training summary: {e}")

    # 2. Generate Test Summary CSV
    try:
        df_test = pd.DataFrame(test_results)
        test_filepath = os.path.join(output_dir, 'test_summary.csv')
        df_test.to_csv(test_filepath, index=False)
        print(f"Successfully saved test summary to: {test_filepath}")
    except Exception as e:
        print(f"Error saving test summary: {e}")

    # 3. Generate Detailed Trade Log for one test episode
    try:
        log_data = _run_test_and_log_trades(agent, env)
        df_log = pd.DataFrame(log_data)
        
        # Reorder columns for clarity
        asset_cols = sorted([col for col in df_log.columns if 'action_' in col or 'value_' in col])
        core_cols = ['timestamp', 'step', 'net_worth', 'cash_balance', 'reward']
        df_log = df_log[core_cols + asset_cols]
        
        log_filepath = os.path.join(output_dir, 'trade_log.csv')
        df_log.to_csv(log_filepath, index=False)
        print(f"Successfully saved detailed trade log to: {log_filepath}")
    except Exception as e:
        print(f"Error saving detailed trade log: {e}")

    print("\n" + "="*60)
    print("All CSV files generated successfully!")
    print("="*60)


def _run_test_and_log_trades(agent, env):
    """
    Runs a single test episode and logs detailed information at each step.

    Returns:
        list: A list of dictionaries, where each dictionary is a log entry for a step.
    """
    print("\nRunning a detailed test episode to generate trade log...")
    log_data = []
    state, info = env.reset()
    done = False
    step = 0

    while not done:
        action = agent.act(state, add_noise=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        log_entry = {
            'timestamp': info.get('timestamp'),
            'step': step,
            'net_worth': info.get('net_worth'),
            'cash_balance': info.get('balance'),
            'reward': reward,
        }

        # Add asset-specific actions and portfolio values
        portfolio_composition = info.get('portfolio_composition', {})
        for i, asset in enumerate(env.asset_names):
            log_entry[f'action_{asset}'] = action[i]
            log_entry[f'value_{asset}'] = portfolio_composition.get(asset, 0)
            
        log_data.append(log_entry)

        state = next_state
        step += 1
    
    print(f"   ...detailed log captured over {step} steps.")
    return log_data

#----------------------------------------------------------------------

if __name__ == "__main__":
    trained_agent, training_results = train_agent()

    if trained_agent is not None and training_results is not None:
        agent = trained_agent
        env = agent.env  # Get the environment from the agent
        net_worths, episode_returns = training_results
        training_completed = True

        print(f"\nFINAL TRAINING RESULTS:")
        print(f"   Final Net Worth: ${net_worths[-1]:,.2f}")
        print(f"   Total Return: {episode_returns[-1]:.2f}%")
        # ... (rest of your print statements)

        # ---- NEW CODE TO RUN TESTS AND GENERATE CSVs ----
        
        # 1. Run the standard test to get summary results
        print("\nRunning agent tests...")
        test_results = test_agent(agent, env, episodes=10, verbose=False) # Run a few test episodes
        
        # 2. Generate all CSV files using the results
        generate_csv_files(
            agent=agent,
            env=env,
            training_results=training_results,
            test_results=test_results
        )
        # --------------------------------------------------

    else:
        print(" Training failed.")
        training_completed = False
