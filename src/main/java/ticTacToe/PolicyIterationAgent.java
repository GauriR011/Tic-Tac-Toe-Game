package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialize a learning agent with default MDP parameters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initializes the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
		
		//creating a variable of Random class
		Random r = new Random();
		
		//setting initial policy
		
		//iterating through each game
		for (Map.Entry<Game, Double> game : policyValues.entrySet()) {
			if(game.getKey().isTerminal()) {
				continue;
			}
			
			//getting all possible moves of game
			List<Move> allMoves = game.getKey().getPossibleMoves();

			//choosing a random move for the game
			Move random = allMoves.get(r.nextInt(allMoves.size()));
			
			//mapping game with a random move and adding to curPolicy 
			curPolicy.put(game.getKey(), random);
			
			
		}
		
		
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the current policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		
		/* YOUR CODE HERE */
		
		//list to store change in values of states/games
		List<Double> valChange = new ArrayList<Double>();
		
		//list to store transitions of each move
		List<TransitionProb> tp = new ArrayList<TransitionProb>();
		double max_val_change = 1;
		
		// perform policy evaluation steps until maximum change in values is greater than delta
		while (max_val_change > delta) {
			valChange.clear();
			
			//iterating through each game
			for (Map.Entry<Game, Double> game : policyValues.entrySet()) {
				
				if(game.getKey().isTerminal()) {
					policyValues.replace(game.getKey(), 0.0);
					continue;
				}
				
				//get move of the game from current policy
				Move m = curPolicy.get(game.getKey());
				
				tp.clear();
				
				//get possible transitions of move
				tp = mdp.generateTransitions(game.getKey(), m);
				
				double val=0;
				double valdifference = 0;
				for (TransitionProb t_item : tp) {
					val+= t_item.prob * (t_item.outcome.localReward + discount * policyValues.get(t_item.outcome.sPrime));
				}
				
				//calculate the difference in values of game state
				valdifference = Math.abs(val - policyValues.get(game.getKey())) ;
				
				//adding the difference to the list
				valChange.add(valdifference);
				
				//update game state value
				policyValues.replace(game.getKey(), val);
			}
			
			
			max_val_change = Collections.max(valChange);
			
		    
		}

		
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		
		/* YOUR CODE HERE */
		
		List<TransitionProb> tp1 = new ArrayList<TransitionProb>();
		HashMap<Game, Move> oldPolicy=new HashMap<Game, Move>();
		Map<Move, Double> maxValue_move = new HashMap<Move, Double>();
		
		//copying current policy to old policy
		for(Map.Entry<Game, Move> g : curPolicy.entrySet()) {
			if(oldPolicy.containsKey(g.getKey())) {
				oldPolicy.replace(g.getKey(), g.getValue());
			}
			else {
				oldPolicy.put(g.getKey(), g.getValue());
			}
			
		}
		
		
		
		//iterating through each game
		for (Map.Entry<Game, Double> item : policyValues.entrySet()) {
			
			//getting all possible moves of game
			List<Move> allMoves = item.getKey().getPossibleMoves();
			maxValue_move.clear();
			
			//calculating Q values
			for(Move m : allMoves) {
				
				tp1.clear();
				tp1 = mdp.generateTransitions(item.getKey(), m);
				
				double val=0;
				for (TransitionProb t_item : tp1) {

					//calculate q values
					//QValue = Transition (Reward + Discount*Value_of_destination_state)
					val += t_item.prob * ( t_item.outcome.localReward + discount*policyValues.get(t_item.outcome.sPrime));	
				}
				
				//adding Q values to a list
				if(maxValue_move.containsKey(m)) {
					maxValue_move.replace(m, val);
				}
				//else, add to map
				else {
					maxValue_move.put(m, val);
				}
			}
			
			//get max Q value and the respective move
			double max_value = -100;
			Move max_move = null;
			for(Map.Entry<Move,Double> entry : maxValue_move.entrySet()) {
				double iter_val = entry.getValue();
				Move iter_key = entry.getKey();
				
				if(iter_val > max_value) {
					max_value = iter_val;
					max_move = iter_key;
				}
			}
			
			//updating the policy
			curPolicy.replace(item.getKey(), max_move);
			
			
		}
		
		//checking if the policy changed
		if(oldPolicy.equals(curPolicy)) {
			return false;
		}
		
		return true;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		/* YOUR CODE HERE */
		
		//running policy evaluation and improve policy functions until the policy converges
		this.evaluatePolicy(delta);
		
		while(this.improvePolicy()) {
			
			 this.evaluatePolicy(delta);
		}
		
		Policy p = new Policy(this.curPolicy);
		super.policy = p;
		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent against a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
