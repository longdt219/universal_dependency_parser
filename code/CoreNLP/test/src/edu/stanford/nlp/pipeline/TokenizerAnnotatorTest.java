package edu.stanford.nlp.pipeline;

import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import junit.framework.TestCase;


/** 
 * See TokenizerAnnotatorITest for some tests that require model files
 *
 * @author Christopher Manning 
 */
public class TokenizerAnnotatorTest extends TestCase {

  private static final String text = "She'll prove it ain't so.";
  private static List<String> tokenWords = Arrays.asList(new String[] {
          "She",
          "'ll",
          "prove",
          "it",
          "ai",
          "n't",
          "so",
          "."
  });

  public void testNewVersion() {
    Annotation ann = new Annotation(text);
    Annotator annotator = new TokenizerAnnotator("en");
    annotator.annotate(ann);
    Iterator<String> it = tokenWords.iterator();
    for (CoreLabel word : ann.get(CoreAnnotations.TokensAnnotation.class)) {
      assertEquals("Bung token in new CoreLabel usage", it.next(), word.word());
    }
    assertFalse("Too few tokens in new CoreLabel usage", it.hasNext());

    Iterator<String> it2 = tokenWords.iterator();
    for (CoreLabel word : ann.get(CoreAnnotations.TokensAnnotation.class)) {
      assertEquals("Bung token in new CoreLabel usage", it2.next(), word.get(CoreAnnotations.TextAnnotation.class));
    }
    assertFalse("Too few tokens in new CoreLabel usage", it2.hasNext());
  }

  public void testBadLanguage() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize");
    props.setProperty("tokenize.language", "notalanguage");
    try {
      StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
      throw new RuntimeException("Should have failed");
    } catch (IllegalArgumentException e) {
      // yay, passed
    }
  }

  public void testDefaultNoNLsPipeline() {
    String t = "Text with \n\n a new \nline.";
    List<String> tWords = Arrays.asList(new String[] {
        "Text",
        "with",
        "a",
        "new",
        "line",
        "."
    });

    Properties props = new Properties();
    props.setProperty("annotators", "tokenize");
    Annotation ann = new Annotation(t);
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    pipeline.annotate(ann);
    Iterator<String> it = tWords.iterator();
    for (CoreLabel word : ann.get(CoreAnnotations.TokensAnnotation.class)) {
      assertEquals("Bung token in new CoreLabel usage", it.next(), word.word());
    }
    assertFalse("Too few tokens in new CoreLabel usage", it.hasNext());

    Iterator<String> it2 = tWords.iterator();
    for (CoreLabel word : ann.get(CoreAnnotations.TokensAnnotation.class)) {
      assertEquals("Bung token in new CoreLabel usage", it2.next(), word.get(CoreAnnotations.TextAnnotation.class));
    }
    assertFalse("Too few tokens in new CoreLabel usage", it2.hasNext());
  }
}
