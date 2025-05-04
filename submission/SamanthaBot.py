import random
import numpy as np
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from collections import defaultdict
import pickle
import os

def setup_ai():
    return SamanthaBot()

class SamanthaBot(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        # Q-learning parameters
        self.q_table = defaultdict(lambda: np.zeros(3))  # [fold, call, raise]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.last_state = None
        self.last_action = None
        # Save Q-table to persist learning
        self.q_table_file = "poker_q_table.pkl"
        self.load_q_table()

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(3), pickle.load(f))

    def get_state(self, hole_card, round_state):
        """Generate a simplified state representation."""
        # Estimate hand strength
        win_rate = self.estimate_hand_strength(hole_card, round_state['community_card'])
        # Discretize pot size
        pot_size = round_state['pot']['main']['amount']
        pot_category = min(pot_size // 50, 3)  # 0-50, 50-100, 100-150, 150+
        # Street encoding
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street = street_map[round_state['street']]
        # Discretize win rate
        strength_category = int(win_rate * 4)  # 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1
        return (strength_category, pot_category, street)

    def estimate_hand_strength(self, hole_card, community_card):
        """Estimate win rate of hole cards using pypokerengine's utility."""
        hole = gen_cards(hole_card)
        community = gen_cards(community_card)
        return estimate_hole_card_win_rate(nb_simulation=100, hole_card=hole, community_card=community)

    def choose_action(self, state, valid_actions):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: choose random valid action
            return random.choice([i for i, a in enumerate(valid_actions) if a['action'] in ['fold', 'call', 'raise']])
        else:
            # Exploitation: choose best action from Q-table
            q_values = self.q_table[state]
            # Only consider valid actions
            valid_indices = []
            for i, action in enumerate(valid_actions):
                if action['action'] == 'fold':
                    valid_indices.append(0)
                elif action['action'] == 'call':
                    valid_indices.append(1)
                elif action['action'] == 'raise':
                    valid_indices.append(2)
            # Mask invalid actions
            masked_q_values = np.array([q_values[i] if i in valid_indices else -np.inf for i in range(3)])
            return np.argmax(masked_q_values)

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action_idx = self.choose_action(state, valid_actions)
        action_map = {0: 'fold', 1: 'call', 2: 'raise'}
        action = action_map[action_idx]

        # Store state and action for Q-table update
        self.last_state = state
        self.last_action = action_idx

        if action == 'fold':
            return self.do_fold(valid_actions)
        elif action == 'call':
            return self.do_call(valid_actions)
        elif action == 'raise':
            # Calculate raise amount (e.g., 2x current pot or min raise)
            min_raise = valid_actions[2]['amount']['min']
            max_raise = valid_actions[2]['amount']['max']
            pot_size = round_state['pot']['main']['amount']
            raise_amount = min(max(min_raise, pot_size * 2), max_raise)
            return self.do_raise(valid_actions, raise_amount)

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Update Q-table based on round outcome."""
        if self.last_state is None or self.last_action is None:
            return

        # Calculate reward
        my_uuid = self.uuid
        my_seat = next(seat for seat in round_state['seats'] if seat['uuid'] == my_uuid)
        final_stack = my_seat['stack']
        # Simple reward: positive if we won, negative if we lost, scaled by pot size
        reward = 0
        if any(winner['uuid'] == my_uuid for winner in winners):
            reward = round_state['pot']['main']['amount']
        else:
            reward = -round_state['pot']['main']['amount'] * 0.1  # Smaller penalty for losing

        # Update Q-value
        current_q = self.q_table[self.last_state][self.last_action]
        # Since it's the end of the round, no next state
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[self.last_state][self.last_action] = new_q

        # Save Q-table
        self.save_q_table()

        # Reset for next round
        self.last_state = None
        self.last_action = None

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    # Helper functions
    def do_fold(self, valid_actions):
        action_info = valid_actions[0]
        amount = action_info["amount"]
        return action_info['action'], amount

    def do_call(self, valid_actions):
        action_info = valid_actions[1]
        amount = action_info["amount"]
        return action_info['action'], amount

    def do_raise(self, valid_actions, raise_amount):
        action_info = valid_actions[2]
        amount = max(action_info['amount']['min'], raise_amount)
        return action_info['action'], amount

    def do_all_in(self, valid_actions):
        action_info = valid_actions[2]
        amount = action_info['amount']['max']
        return action_info['action'], amount
