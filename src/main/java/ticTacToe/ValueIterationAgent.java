package ticTacToe;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor (gamma)
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 * k=number of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialize your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues(); //sets initial values of all states
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, 
	 * and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and 
	 * {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. 
	 * 
	 * After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. 
	 *
	 * You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	
	public void iterate()
	{
		/* 
		 * YOUR CODE HERE
		 *
		 */
		
		//List to store transition tuples of each move
		List<TransitionProb> tp = new ArrayList<TransitionProb>();
		
		//Maps the move to the value
		Map<Move, Double> maxValue_move = new HashMap<Move, Double>();
		
		//iterate k times
		for (int i=0; i<k; i++) { 
			
			//iterating through each game in value function
			for (Map.Entry<Game, Double> game : valueFunction.entrySet()) {
				
				//if the game is a terminal state, then assign the value of the state to be 0
				if(game.getKey().isTerminal()) {
					valueFunction.replace(game.getKey(), 0.0);
					continue;
				}
				
				maxValue_move.clear();
				
				//getting all possible moves of game
				List<Move> allMoves = game.getKey().getPossibleMoves();
				
				//iterating through all moves
				for(Move m : allMoves) {
					tp.clear();
					
					//generating transitions for each move
					tp = mdp.generateTransitions(game.getKey(), m);
					
					double val=0;
					//iterating through each transition
					for (TransitionProb t_item : tp) {
						//calculate q values
						//QValue = Transition (Reward + Discount*Value_of_destination_state)
						val += t_item.prob * ( t_item.outcome.localReward + discount*valueFunction.get(t_item.outcome.sPrime));	
					}
					
					//mapping move to value
					//if the move is present in the map, replace it
					if(maxValue_move.containsKey(m)) {
						maxValue_move.replace(m, val);
					}
					//else, add to map
					else {
						maxValue_move.put(m, val);
					}
					
				}
				
				//choosing move with max value
				double max_val = -100;  
				
				//iterating through the q values
				for(Map.Entry<Move,Double> entry : maxValue_move.entrySet()) {
					
					double iter_val = entry.getValue();
					
					if(iter_val > max_val) {
						max_val = iter_val;
						
					}
				}
				
				//update valueFunction with new value
				valueFunction.replace(game.getKey(), max_val);
				
				
			}
		}
			
		
	}
	
	/**This method should be run AFTER the train method to extract a policy according 
	 * to {@link ValueIterationAgent#valueFunction}
	 * valueFunction is a hashmap that stores state-value pairs
	 * 
	 * You will need to do a single step of expectimax from each game (state) key in 
	 * {@link ValueIterationAgent#valueFunction} to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
		
		HashMap<Game, Move> policy=new HashMap<Game, Move>();
		List<TransitionProb> tp = new ArrayList<TransitionProb>();
		Map<Move, Double> maxValue_move = new HashMap<Move, Double>();
		
		//iterating through contents of value function 
		for (Map.Entry<Game, Double> item : valueFunction.entrySet()) {
			
			
			//getting all possible moves of game
			List<Move> allMoves = item.getKey().getPossibleMoves();
			maxValue_move.clear();
			
			//calculating Q values
			for(Move m : allMoves) {
				
				tp.clear();
				tp = mdp.generateTransitions(item.getKey(), m);
				
				double val=0;
				for (TransitionProb t_item : tp) {

					//calculate q values
					//QValue = Transition (Reward + Discount*Value_of_destination_state)
					val += t_item.prob * ( t_item.outcome.localReward + discount*valueFunction.get(t_item.outcome.sPrime));	
				}
				
				//adding the Q values to a list 
				if(maxValue_move.containsKey(m)) {
					maxValue_move.replace(m, val);
				}
				//else, add to map
				else {
					maxValue_move.put(m, val);
				}
			}
			
			
			//choosing the move giving maximum value
			for(Map.Entry<Move,Double> entry : maxValue_move.entrySet()) {
				double iter_val = entry.getValue();
				Move iter_key = entry.getKey();
				
				//mapping the move to the game in the policy
				if(iter_val == item.getValue()) {
					policy.put(item.getKey(), iter_key);
					break;
				}
			}
			
			
		}
		
		return new Policy(policy);

	}
	
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} 
		 * and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		//RandomAgent r=new RandomAgent();
		Game g=new Game(agent, d, d); //edit this line
		g.playOut();
		
		
		

		
		
	}
}
