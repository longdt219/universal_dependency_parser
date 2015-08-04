package edu.stanford.nlp.semgraph;

import edu.stanford.nlp.ling.CoreAnnotation;


/** This class collects CoreAnnotations that are used in working with a
 *  SemanticGraph.  (These were originally separated out at a time when
 *  a SemanticGraph was backed by the JGraphT library so as not to
 *  introduce a library dependency for some tools. This is no longer
 *  the case, but they remain gathered here.)
 *
 *  @author Christopher Manning
 */
public class SemanticGraphCoreAnnotations {

  /**
   * The CoreMap key for getting the syntactic dependencies of a sentence.
   * These are collapsed dependencies!
   *
   * This key is typically set on sentence annotations.
   */
  public static class CollapsedDependenciesAnnotation implements CoreAnnotation<SemanticGraph> {
    @Override
    public Class<SemanticGraph> getType() {
      return SemanticGraph.class;
    }
  }


  /**
   * The CoreMap key for getting the syntactic dependencies of a sentence.
   * These are basic dependencies without any post-processing!
   *
   * This key is typically set on sentence annotations.
   */
  public static class BasicDependenciesAnnotation implements CoreAnnotation<SemanticGraph> {
    @Override
    public Class<SemanticGraph> getType() {
      return SemanticGraph.class;
    }
  }


  /**
   * The CoreMap key for getting the syntactic dependencies of a sentence.
   * These are dependencies that are both collapsed and have CC processing!
   *
   * This key is typically set on sentence annotations.
   */
  public static class CollapsedCCProcessedDependenciesAnnotation implements CoreAnnotation<SemanticGraph> {
    @Override
    public Class<SemanticGraph> getType() {
      return SemanticGraph.class;
    }
  }

}
