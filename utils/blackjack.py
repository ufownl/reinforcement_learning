import random


def deal_card():
    return random.choice(sum([[i] * 4 for i in range(1, 10)], start=[]) + [10] * 16)


class State:
    def __init__(self, usable_ace, player_sum, dealer_card):
        self.__usable_ace = 1 if usable_ace else 0
        if player_sum < 12:
            self.__player_sum = 0
        elif player_sum > 21:
            self.__player_sum = 9
        else:
            self.__player_sum = player_sum - 12
        if dealer_card < 1:
            self.__dealer_card = 0
        elif dealer_card > 10:
            self.__dealer_card = 9
        else:
            self.__dealer_card = dealer_card - 1

    @property
    def usable_ace(self):
        return self.__usable_ace != 0

    @property
    def player_sum(self):
        return self.__player_sum + 12

    @property
    def dealer_card(self):
        return self.__dealer_card + 1

    def index(self, flat=True):
        if flat:
            return self.__usable_ace * 100 + self.__player_sum * 10 + self.__dealer_card
        else:
            return (self.__usable_ace, self.__player_sum, self.__dealer_card)

    def hit(self):
        player_sum = self.player_sum + deal_card()
        if player_sum > 21:
            if self.usable_ace:
                return (State(False, player_sum - 10, self.dealer_card), 0)
            else:
                return (None, -1)
        else:
            return (State(self.usable_ace, player_sum, self.dealer_card), 0)

    def stick(self):
        dealer_sum = 11 if self.dealer_card == 1 else self.dealer_card
        while dealer_sum < 17:
            card = deal_card()
            dealer_sum += 11 if dealer_sum < 11 and card == 1 else card
        if dealer_sum > 21 or self.player_sum > dealer_sum:
            return (None, 1)
        elif self.player_sum == dealer_sum:
            return (None, 0)
        else:
            return (None, -1)
