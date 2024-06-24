package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} 
 * which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent 
 * has played their move. 
 * You may want/need to edit {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=100000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initializes all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 100000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		/* 
		 * YOUR CODE HERE
		 */
		
		for(int i=0; i<this.numEpisodes; i++) {
			//getting current game
			Game g = env.getCurrentGameState();
			
			//running until current game reaches terminal state
			 while(!g.isTerminal()) {
				//choose move randomly
					double val;
					Move game_move = null;
					
					//generating a random number
					double prob = Math.random();
					
					//choosing control policy (exploration / exploitation)
					
					if(prob <= epsilon) {
						//choose random move
						Random r = new Random();
						List<Move> allMoves = g.getPossibleMoves();
						
						//choosing a random move
						Move random = allMoves.get(r.nextInt(allMoves.size()));
						game_move = random;
						
					}
					
					else {
						//choose move from qTable
						Map <Move, Double> max_value_list = new HashMap<Move, Double>();
						max_value_list.clear();
						
						//choose max value move from Q table
						List<Move> moves=g.getPossibleMoves();
						
						for(Move m: moves)
						{
							//getting the Q values and adding to a list
							max_value_list.put(m,qTable.getQValue(g, m));
						}
						
						double max_val = -100;
						Move max_move = null;
						
						//choosing the max Q value and its respective move
						for(Map.Entry<Move, Double> item : max_value_list.entrySet()) {
							if(item.getValue()>max_val) {
								max_val = item.getValue();
								max_move = item.getKey();
							}
						}
						
						game_move = max_move;
						
					}
					
					//calculate the Q value of game
					val = 0;
					double reward = 0;
					Game succ_game = null;
					Outcome outcome;
					
					try {
						//getting the outcome on executing the move
						outcome = env.executeMove(game_move);
						reward = outcome.localReward;
						succ_game = outcome.sPrime;
						
						//if the game reaches a terminal state
					} catch (IllegalMoveException e) {
						// TODO Auto-generated catch block
						qTable.addQValue(succ_game, game_move, 0.0);
						continue;
					}
					
					
					//calculating the Q value
					
					double max_succ_val = -100;
					
					//if the successor game state is terminal, set the value as 0
					if(succ_game.isTerminal()) {
						max_succ_val = 0;
					}
					else {
						List<Move> all_moves = succ_game.getPossibleMoves();
						
						for(Move ele : all_moves) {
							
							//get value of each move
							if(max_succ_val < qTable.getQValue(succ_game, ele)) {
								max_succ_val = qTable.getQValue(succ_game, ele);
							}
							
						}
					}

					val = reward + discount * max_succ_val;
					double val1 = (1-alpha)*qTable.getQValue(outcome.s, game_move) + alpha * val;
					
					//updating the qTable
					qTable.addQValue(outcome.s, game_move, val1);
			 }
			 
			 //resetting the current game state when it reaches a terminal state
			 env.reset();
		}
		
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
		
		super.policy = extractPolicy();
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		/* 
		 * YOUR CODE HERE
		 */
		HashMap<Game, Move> policy=new HashMap<Game, Move>();
		Map <Move, Double> max_value_list = new HashMap<Move, Double>();
		
		//generating all valid games
		List<Game> allGames=Game.generateAllValidGames('X');
		
		//iterating through each game
		for(Game g: allGames){
			max_value_list.clear();
			
			//getting all possible moves
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				//getting the Q values and adding to a list
				max_value_list.put(m,qTable.getQValue(g, m));
			}
			
			double max_val = -100;
			Move max_move = null;
			
			//choosing the max Q value and its respective move
			for(Map.Entry<Move, Double> item : max_value_list.entrySet()) {
				if(item.getValue()>max_val) {
					max_val = item.getValue();
					max_move = item.getKey();
				}
			}
			
			//mapping the move to the game in the policy
			policy.put(g, max_move);
			
		}
		
		return new Policy(policy);
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
	}
	
}
