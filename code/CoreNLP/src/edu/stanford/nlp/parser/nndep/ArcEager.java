package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.CoreMap;

import java.util.ArrayList;
import java.util.List;

/**
 * Defines an arc-eager transition-based dependency parsing system
 * (Nivre, 2005).
 *
 * @author Long Duong 
 */
public class ArcEager extends ParsingSystem {
  private boolean singleRoot = true;

  public ArcEager(TreebankLanguagePack tlp, List<String> labels, boolean verbose) {
    super(tlp, labels, verbose);
  }

  @Override
  public boolean isTerminal(Configuration c) {
    return c.getBufferSize() == 0;
  }

  @Override
  public void makeTransitions() {
    transitions = new ArrayList<>();
    // Long Duong : Left arc, Right arc and shift 
    // TODO store these as objects!
    for (String label : labels)
      transitions.add("L(" + label + ")");
    for (String label : labels)
      transitions.add("R(" + label + ")");

    transitions.add("S");
    // Long Duong : Add reduce parsing transition
    transitions.add("U");
  }

  @Override
  public Configuration initialConfiguration(CoreMap s) {
    Configuration c = new Configuration(s);
    int length = s.get(CoreAnnotations.TokensAnnotation.class).size();

    // For each token, add dummy elements to the configuration's tree
    // and add the words onto the buffer
    for (int i = 1; i <= length; ++i) {
      c.tree.add(Config.NONEXIST, Config.UNKNOWN);
      c.buffer.add(i);
    }

    // Put the ROOT node on the stack
    c.stack.add(0);

    return c;
  }

  @Override
  public boolean canApply(Configuration c, String t) {
    if (t.startsWith("L") || t.startsWith("R")) 
    	if ((c.getStackSize() <= 0) || (c.getBufferSize() <=0)) return false;
    
    if (t.startsWith("L")){
    	if (c.getStack(0) == 0) return false; // first Item of stack must be not the root
    	int wi = c.getStack(0); 
    	int wj = c.getBuffer(0); 
    	if (c.getHead(wi) != Config.NONEXIST) return false; // Already added to A 
    }
    if (t.startsWith("U")){  // Reduce
    	if (c.getStackSize() <= 0) return false; 
    	int wi = c.getStack(0);
    	if (c.getHead(wi) == Config.NONEXIST) return false; // Not in A 
    } 
    if (t.startsWith("S"))
    	if (c.getBufferSize() <=0) return false; 
    
    return true;  
  }

  @Override
  public void apply(Configuration c, String t) {
    if (t.startsWith("L")) {
    	int wi = c.getStack(0); 
    	int wj = c.getBuffer(0);
    	String label = t.substring(2, t.length() - 1);
    	c.addArc(wj, wi, label);
    	c.removeTopStack();
    	return ;
    }
    
    if (t.startsWith("R")) {
    	int wi = c.getStack(0); 
    	int wj = c.getBuffer(0);    	
    	String label = t.substring(2, t.length() - 1);
    	c.addArc(wi, wj, label);
    	c.removeFirstBuffer();
    	c.addTopStack(wj);
    	return; 
    } 
    
    if (t.startsWith("U")){ // Reduce 
    	c.removeTopStack();
    	return;
    }
    if (t.startsWith("S"))
    	c.shift();
    else
    	System.out.println(" Long Duong : Expect Erorr when transition is not matched ");
  }
  
  // O(n) implementation
  @Override
  public String getOracle(Configuration c, DependencyTree dTree) {
	  if ((c.getStackSize()>0) && (c.getBufferSize() >0)){
		  int wi = c.getStack(0); 
		  int wj = c.getBuffer(0);
		  if (dTree.getHead(wi) == wj)
			  return "L(" + dTree.getLabel(wi) + ")";
		  
		  if (dTree.getHead(wj) == wi)
			  return "R(" + dTree.getLabel(wj) + ")";
		  
		  boolean flag = false; 
		  for (int k = 0; k<wi; k++){
			  if (dTree.getHead(wj) == k) flag = true; 
			  if (dTree.getHead(k) == wj) flag = true; 
		  }
		  if (flag){
			  return "U";
		  }
	  } 
      return "S";
  }

@Override
boolean isOracle(Configuration c, String t, DependencyTree dTree) {
	// TODO Auto-generated method stub
	return false;
}

}
