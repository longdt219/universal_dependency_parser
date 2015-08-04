package edu.stanford.nlp.ie.crf;

import junit.framework.TestCase;

import java.util.Properties;

/** 
 * Test that the CRFClassifier works when multiple classifiers are run
 * in multiple threads.
 *
 *  @author John Bauer
 */
public class ThreadedCRFClassifierITest extends TestCase {
  Properties props;

  private String german1 = 
    "/u/nlp/data/ner/goodClassifiers/german.hgc_175m_600.crf.ser.gz";
  private String german2 = 
    "/u/nlp/data/ner/goodClassifiers/german.dewac_175m_600.crf.ser.gz";
  private String germanTestFile = "/u/nlp/data/german/ner/deu.testa";

  private String english1 = 
    "/u/nlp/data/ner/goodClassifiers/english.all.3class.nodistsim.crf.ser.gz";
  private String english2 = 
    "/u/nlp/data/ner/goodClassifiers/english.all.3class.distsim.crf.ser.gz";
  private String englishTestFile = "/u/nlp/data/ner/column_data/conll.testa";

  private String germanEncoding = "iso-8859-1";
  private String englishEncoding = "utf-8";
  
  @Override
  public void setUp() {
    props = new Properties();
  }

  public void testOneEnglishCRF() {
    props.setProperty("crf1", english1);
    props.setProperty("testFile", englishTestFile);
    props.setProperty("inputEncoding", englishEncoding);
    TestThreadedCRFClassifier.runTest(props);
  }

  public void testOneGermanCRF() {
    props.setProperty("crf1", german1);
    props.setProperty("testFile", germanTestFile);
    props.setProperty("inputEncoding", germanEncoding);
    TestThreadedCRFClassifier.runTest(props);
  }

  public void testTwoGermanCRFs() {
    props.setProperty("crf1", german1);
    props.setProperty("crf2", german2);
    props.setProperty("testFile", germanTestFile);
    props.setProperty("inputEncoding", germanEncoding);
    TestThreadedCRFClassifier.runTest(props);
  }
}

