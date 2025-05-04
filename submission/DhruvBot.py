import random
from pypokerengine.players import BasePokerPlayer

def setup_ai():
    return DhruvBot()

class DhruvBot(BasePokerPlayer):

    def __init__(self):
        super().__init__()

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        street = round_state['street']
        my_uuid = self.uuid
        seats = round_state['seats']
        my_stack = next(seat for seat in seats if seat['uuid'] == my_uuid)['stack']

        hole_ranks = [card[0] for card in hole_card]
        high_cards = ['A', 'K', 'Q', 'J', 'T']

        if street == 'preflop':
            # --- Add randomness for bluffing/folding ---
            rand = random.random()
            if rand < 0.13:
                return self.do_all_in(valid_actions)  # 13% bluff all-in
            elif rand < 0.25:
                return self.do_fold(valid_actions)    # Next 12% chance: fold junk

            # --- Evaluate hand strength ---
            is_pair = hole_ranks[0] == hole_ranks[1]
            high_card_count = sum([1 for r in hole_ranks if r in high_cards])

            if is_pair and hole_ranks[0] in high_cards:
                return self.do_all_in(valid_actions)  # Strong pair

            elif high_card_count == 2:
                return self.do_raise(valid_actions, 100)  # Two high cards

            elif high_card_count == 1:
                return self.do_call(valid_actions)

            else:
                return self.do_fold(valid_actions)

        # Post-flop strategy: 70% call, 30% fold (can be improved)
        if community_card:
            return self.do_call(valid_actions) if random.random() < 0.7 else self.do_fold(valid_actions)

        return self.do_call(valid_actions)

    # Game state methods
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

    # Action helpers
    def do_fold(self, valid_actions):
        return valid_actions[0]['action'], valid_actions[0]['amount']

    def do_call(self, valid_actions):
        return valid_actions[1]['action'], valid_actions[1]['amount']

    def do_raise(self, valid_actions, raise_amount):
        action_info = valid_actions[2]
        amount = max(action_info['amount']['min'], min(raise_amount, action_info['amount']['max']))
        return action_info['action'], amount

    def do_all_in(self, valid_actions):
        action_info = valid_actions[2]
        return action_info['action'], action_info['amount']['max']