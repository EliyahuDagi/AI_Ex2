"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name:
Student ID:

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math
import sys

import numpy as np

import gameUtil
from connect4 import GameState
from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @staticmethod
    def play(gameState: GameState, action: int) -> GameState:
        next_state = gameState.generateSuccessor(gameState.get_piece_player(), action)
        next_state.switch_turn(next_state.turn)
        return next_state

class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """

        actions = gameState.getLegalActions()
        actions_gain = []
        for action in actions:
            actions_gain.append(self.min_max_value(self.play(gameState, action), 1))
        max_action_idx = np.argmax(actions_gain)
        return actions[max_action_idx]

    def min_max_value(self, gameState: GameState, cur_depth):
        if gameState.is_terminal() or cur_depth >= self.depth:
            return self.evaluationFunction(gameState)
        func = min if gameState.turn == 0 else max
        actions_gain = [self.min_max_value(self.play(gameState, action), cur_depth=cur_depth + 1)
                        for action in gameState.getLegalActions()]
        chosen_gain = func(actions_gain)
        return chosen_gain


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """
        actions = gameState.getLegalActions()
        actions_gain = []
        for action in actions:
            actions_gain.append(self.min_value(self.play(gameState, action), 1, float('-inf'), float('inf')))
        max_action_idx = np.argmax(actions_gain)
        return actions[max_action_idx]

    def min_value(self, gameState: GameState, cur_depth, alpha, beta):
        if gameState.is_terminal() or cur_depth >= self.depth:
            return self.evaluationFunction(gameState)
        min_val = float('inf')
        for action in gameState.getLegalActions():
            cur_gain = self.max_value(self.play(gameState, action), alpha=alpha, beta=beta, cur_depth=cur_depth + 1)
            min_val = min(cur_gain, min_val)
            if cur_gain < alpha:
                return cur_gain
            beta = min(cur_gain, beta)
        return min_val

    def max_value(self, gameState: GameState, cur_depth, alpha, beta):
        if gameState.is_terminal() or cur_depth >= self.depth:
            return self.evaluationFunction(gameState)
        max_val = float('-inf')
        for action in gameState.getLegalActions():
            cur_gain = self.min_value(self.play(gameState, action), alpha=alpha, beta=beta, cur_depth=cur_depth + 1)
            max_val = max(cur_gain, max_val)
            if cur_gain > beta:
                return cur_gain
            alpha = max(cur_gain, alpha)
        return max_val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
