import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 2
        self.action_size = 4
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 0
        self.epsilon_decay = 0.5 # original :0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.gamma = 0  # discount rate
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)
        self.day = 0

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.vote_num = {
            Vote.SELL: 0,
            Vote.HOLD: 1,
            Vote.BUY: 2,
        }

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def decide_action(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        # get actions and best action
        action_options = self.model.predict(state)
        best_action = np.argmax(action_options[0])
        return best_action

    def train_network(self, batch_size):
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
            state = np.array(minibatch)[:,0]
            action = np.array(minibatch)[:, 1]
            reward = np.array(minibatch)[:, 2]
            next_state = np.array(minibatch)[:, 3]

            state = [np.append(np.array([]), s) for s in state]
            state = np.array(state)

            next_state = [np.append(np.array([]), ns) for ns in next_state]
            next_state = np.array(next_state)

            pred_next = self.model.predict(next_state, batch_size=self.batch_size)
            target_f = self.model.predict(state, batch_size=self.batch_size)
            for idx, t in enumerate(target_f):
                t[action[idx]] = reward[idx] + self.gamma * np.amax(pred_next[idx])
            self.model.fit(state, target_f, epochs=1, verbose=0)


    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]
        self.day += 1
        # TODO Compute the current state
        order_list = []
        stock_data_a = stock_market_data[Company.A]
        stock_data_b = stock_market_data[Company.B]
        # Expert A
        expert_a = self.expert_a.vote(stock_data_a)
        #  Expert B
        expert_b = self.expert_b.vote(stock_data_b)

        state = np.array([[
            self.vote_num[expert_a], self.vote_num[expert_b], ]])


        action = self.decide_action(state)

        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once

        if self.last_state is not None:
            reward = (portfolio.get_value(stock_market_data) - self.last_portfolio_value) / self.last_portfolio_value
            self.memory.append((self.last_state, self.last_action, reward, state))
            self.train_network(self.batch_size)

        # TODO Create actions for current state and decrease epsilon for fewer random actions

        act0 = 0
        act1 = 0
        act2 = 0
        act3 = 0
        act4 = 0
        act5 = 0
        act6 = 0
        act7 = 0
        act8 = 0

        # What amount of the stocks should be bought or sold
        percent_buy = 1
        percent_sell = 1

        if action == 0:
            # Buy A
            stock_price_a = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int((portfolio.cash*percent_buy/2) // stock_price_a)
            if amount_to_buy_a > 0:
                order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_a))
            # Buy B
            stock_price_b = stock_market_data.get_most_recent_price(Company.B)
            amount_to_buy_b = int((portfolio.cash*percent_buy/2) // stock_price_b)
            if amount_to_buy_b > 0:
                order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_b))
            act0 += 1
        elif action == 1:
            # Buy A
            stock_price_a = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int(portfolio.cash *percent_buy// stock_price_a)
            if amount_to_buy_a > 0:
                order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_a))
            # Sell B
            amount_to_sell_b = int(portfolio.get_stock(Company.B)*percent_sell)
            if amount_to_sell_b > 0:
                order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_b))
            act1 += 1
        elif action == 2:
            # Sell A
            amount_to_sell_a = int(portfolio.get_stock(Company.A)*percent_sell)
            if amount_to_sell_a > 0:
                order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_a))
            # Buy B
            stock_price_b = stock_market_data.get_most_recent_price(Company.B)
            amount_to_buy_b = int(portfolio.cash*percent_buy // stock_price_b)
            if amount_to_buy_b > 0:
                order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_b))
            act2 += 1
        elif action == 3:
            # Sell A
            amount_to_sell_a = int(portfolio.get_stock(Company.A)*percent_sell)
            if amount_to_sell_a > 0:
                order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_a))
            # Sell B
            amount_to_sell_b = int(portfolio.get_stock(Company.B)*percent_sell)
            if amount_to_sell_b > 0:
                order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_b))
            act3 += 1
        elif action == 4:
            # Sell A
            amount_to_sell_a = int(portfolio.get_stock(Company.A)*percent_sell)
            if amount_to_sell_a > 0:
                order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell_a))
            # Hold B
            act4 += 1
        elif action == 5:
            # Hold A
            # Sell B
            amount_to_sell_b = int(portfolio.get_stock(Company.B)*percent_sell)
            if amount_to_sell_b > 0:
                order_list.append(Order(OrderType.SELL, Company.B, amount_to_sell_b))
            act5 += 1
        elif action == 6:
            # Buy A
            stock_price_a = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int((portfolio.cash*percent_buy) // stock_price_a)
            if amount_to_buy_a > 0:
                order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_a))
            # Hold B
            act6 += 1
        elif action == 7:
            # Hold A
            # Buy B
            stock_price_b = stock_market_data.get_most_recent_price(Company.B)
            amount_to_buy_b = int((portfolio.cash*percent_buy) // stock_price_b)
            if amount_to_buy_b > 0:
                order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_b))
            act7 += 1
        elif action == 8:
            # Hold A
            # Hold B
            order_list.append(Order(OrderType.BUY, Company.B, 0))
            act8 += 1
        else:
            print("undefined action called"+str(action))


        # Decrease the epsilon for fewer random actions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # TODO Save created state, actions and portfolio value for the next call of trade()

        self.last_state = state
        self.last_action = action
        self.last_portfolio_value = portfolio.get_value(stock_market_data)

        return order_list


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()

