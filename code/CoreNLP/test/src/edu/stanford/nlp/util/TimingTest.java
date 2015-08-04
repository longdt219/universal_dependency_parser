package edu.stanford.nlp.util;

import java.util.Locale;

import junit.framework.TestCase;

/**
 * @author Christopher Manning
 */
public class TimingTest extends TestCase {

  @Override
  protected void setUp() {
    Locale.setDefault(Locale.US);
  }

  /** There's a lot of time slop in these tests so they don't fire by mistake.
   *  You definitely get them more than 30% off sometimes. :(
   *  And then we got a test failure that was over 70% off on the first test. :(
   *  So, really this only tests that the answers are right to an order of magnitude.
   */
  public void testTiming() {
    Timing t = new Timing();
    sleepTen();
    long val2 = t.reportNano();
    assertEquals("Wrong nanosleep", 10_000_000, val2, 10_000_000);
    // System.err.println(val2);
    sleepTen();
    long val = t.report();
    // System.err.println(val);
    assertEquals("Wrong sleep", 20, val, 10);
    for (int i = 0; i < 8; i++) {
      sleepTen();
    }
    long val3 = t.report();
    assertEquals("Wrong formatted time", "0.1", Timing.toSecondsString(val3));
  }

  private static void sleepTen() {
    try {
      Thread.sleep(10);
    } catch (InterruptedException ie) {
      // do nothing
    }
  }

}