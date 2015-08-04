package edu.stanford.nlp.pipeline;

import edu.stanford.nlp.ie.NERClassifierCombiner;
import edu.stanford.nlp.ie.regexp.NumberSequenceClassifier;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.RuntimeInterruptedException;
import edu.stanford.nlp.util.Timing;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

/**
 * This class will add NER information to an
 * Annotation using a combination of NER models.
 * It assumes that the Annotation
 * already contains the tokenized words as a
 * List&lt;? extends CoreLabel&gt; or a
 * List&lt;List&lt;? extends CoreLabel&gt;&gt; under Annotation.WORDS_KEY
 * and adds NER information to each CoreLabel,
 * in the CoreLabel.NER_KEY field.  It uses
 * the NERClassifierCombiner class in the ie package.
 *
 * @author Jenny Finkel
 * @author Mihai Surdeanu (modified it to work with the new NERClassifierCombiner)
 */
public class NERCombinerAnnotator extends SentenceAnnotator {

  private final NERClassifierCombiner ner;

  private final Timing timer = new Timing();
  private boolean VERBOSE = true;

  private final long maxTime;
  private final int nThreads;

  public NERCombinerAnnotator() throws IOException, ClassNotFoundException {
    this(true);
  }

  private void timerStart(String msg) {
    if(VERBOSE){
      timer.start();
      System.err.println(msg);
    }
  }
  private void timerStop() {
    if(VERBOSE){
      timer.stop("done.");
    }
  }

  public NERCombinerAnnotator(boolean verbose)
    throws IOException, ClassNotFoundException
  {
    this(new NERClassifierCombiner(new Properties()), verbose);
  }

  public NERCombinerAnnotator(boolean verbose, String... classifiers)
    throws IOException, ClassNotFoundException
  {
    this(new NERClassifierCombiner(classifiers), verbose);
  }

  public NERCombinerAnnotator(NERClassifierCombiner ner, boolean verbose) {
    this(ner, verbose, 1, 0);
  }

  public NERCombinerAnnotator(NERClassifierCombiner ner, boolean verbose, int nThreads, long maxTime) {
    VERBOSE = verbose;
    this.ner = ner;
    this.maxTime = maxTime;
    this.nThreads = nThreads;
  }

  public NERCombinerAnnotator(String name, Properties properties) {
    this(createNERClassifierCombiner(name, properties), false,
         PropertiesUtils.getInt(properties, name + ".nthreads", PropertiesUtils.getInt(properties, "nthreads", 1)),
         PropertiesUtils.getLong(properties, name + ".maxtime", -1));
  }

  final static NERClassifierCombiner createNERClassifierCombiner(String name, Properties properties) {
    // TODO: Move function into NERClassifierCombiner?
    List<String> models = new ArrayList<String>();
    String prefix = (name != null)? name + ".": "ner.";
    String modelNames = properties.getProperty(prefix + "model");
    if (modelNames == null) {
      modelNames = DefaultPaths.DEFAULT_NER_THREECLASS_MODEL + "," + DefaultPaths.DEFAULT_NER_MUC_MODEL + "," + DefaultPaths.DEFAULT_NER_CONLL_MODEL;
    }
    if (modelNames.length() > 0) {
      models.addAll(Arrays.asList(modelNames.split(",")));
    }
    if (models.isEmpty()) {
      // Allow for no real NER model - can just use numeric classifiers or SUTime
      System.err.println("WARNING: no NER models specified");
    }
    NERClassifierCombiner nerCombiner;
    try {
      // TODO: use constants for part after prefix so we can ensure consistent options
      boolean applyNumericClassifiers =
              PropertiesUtils.getBool(properties,
                      prefix + "applyNumericClassifiers",
                      NERClassifierCombiner.APPLY_NUMERIC_CLASSIFIERS_DEFAULT);
      boolean useSUTime =
              PropertiesUtils.getBool(properties,
                      prefix + "useSUTime",
                      NumberSequenceClassifier.USE_SUTIME_DEFAULT);
      // TODO: properties are passed in as it for number sequence classifiers (don't care about the prefix)
      nerCombiner = new NERClassifierCombiner(applyNumericClassifiers,
              useSUTime, properties,
              models.toArray(new String[models.size()]));
    } catch (FileNotFoundException e) {
      throw new RuntimeIOException(e);
    }

    return nerCombiner;
  }

  @Override
  protected int nThreads() {
    return nThreads;
  }

  @Override
  protected long maxTime() {
    return maxTime;
  };

  @Override
  public void annotate(Annotation annotation) {
    timerStart("Adding NER Combiner annotation...");

    super.annotate(annotation);

    this.ner.finalizeAnnotation(annotation);
    timerStop();
  }

  @Override
  public void doOneSentence(Annotation annotation, CoreMap sentence) {
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    List<CoreLabel> output = null;
    try {
      output = this.ner.classifySentenceWithGlobalInformation(tokens, annotation, sentence);
    } catch (RuntimeInterruptedException e) {
      // If we get interrupted, set the NER labels to the background
      // symbol if they are not already set, then exit.
      doOneFailedSentence(annotation, sentence);
      return;
    }
    if (VERBOSE) {
      boolean first = true;
      System.err.print("NERCombinerAnnotator direct output: [");
      for (CoreLabel w : output) {
        if (first) { first = false; } else { System.err.print(", "); }
        System.err.print(w.toString());
      }
      System.err.println(']');
    }

    for (int i = 0; i < tokens.size(); ++i) {
      // add the named entity tag to each token
      String neTag = output.get(i).get(CoreAnnotations.NamedEntityTagAnnotation.class);
      String normNeTag = output.get(i).get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class);
      tokens.get(i).setNER(neTag);
      if(normNeTag != null) tokens.get(i).set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, normNeTag);
      NumberSequenceClassifier.transferAnnotations(output.get(i), tokens.get(i));
    }

    if (VERBOSE) {
      boolean first = true;
      System.err.print("NERCombinerAnnotator output: [");
      for (CoreLabel w : tokens) {
        if (first) { first = false; } else { System.err.print(", "); }
        System.err.print(w.toShorterString("Word", "NamedEntityTag", "NormalizedNamedEntityTag"));
      }
      System.err.println(']');
    }
  }

  @Override
  public void doOneFailedSentence(Annotation annotation, CoreMap sentence) {
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    for (int i = 0; i < tokens.size(); ++i) {
      if (tokens.get(i).ner() == null) {
        tokens.get(i).setNER(this.ner.backgroundSymbol());
      }
    }
  }

  @Override
  public Set<Requirement> requires() {
    // TODO: we could check the models to see which ones use lemmas
    // and which ones use pos tags
    if (ner.usesSUTime() || ner.appliesNumericClassifiers()) {
      return TOKENIZE_SSPLIT_POS_LEMMA;
    } else {
      return TOKENIZE_AND_SSPLIT;
    }
  }

  @Override
  public Set<Requirement> requirementsSatisfied() {
    return Collections.singleton(NER_REQUIREMENT);
  }
}
