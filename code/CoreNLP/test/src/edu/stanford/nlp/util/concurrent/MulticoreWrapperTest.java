package edu.stanford.nlp.util.concurrent;

import java.util.Random;

import junit.framework.TestCase;

/**
 * Test of MulticoreWrapper.
 * 
 * @author Spence Green
 *
 */
public class MulticoreWrapperTest extends TestCase {

  private MulticoreWrapper<Integer,Integer> wrapper;
  private int nThreads;
  
  public void setUp() {
    // Automagically detect the number of cores
    nThreads = -1;
  }

  public void testSynchronization() {
    wrapper = new MulticoreWrapper<Integer,Integer>(nThreads, new DelayedIdentityFunction());
    int lastReturned = 0;
    final int nItems = 1000;
    for (int i = 0; i < nItems; ++i) {
      wrapper.put(i);

      while(wrapper.peek()) {
        int result = wrapper.poll();
        System.err.printf("Result: %d%n", result);
        assertEquals(result,lastReturned++);
      }
    }
    
    wrapper.join();
    while(wrapper.peek()) {
      int result = wrapper.poll();
      System.err.printf("Result2: %d%n", result);
      assertEquals(result,lastReturned++);        
    }
  }

  public void testUnsynchronized() {
    wrapper = new MulticoreWrapper<Integer,Integer>(nThreads, new DelayedIdentityFunction(), false);
    int nReturned = 0;
    final int nItems = 1000;
    for (int i = 0; i < nItems; ++i) {
      wrapper.put(i);

      while(wrapper.peek()) {
        int result = wrapper.poll();
        System.err.printf("Result: %d%n", result);
        nReturned++;
      }
    }
    
    wrapper.join();
    while(wrapper.peek()) {
      int result = wrapper.poll();
      System.err.printf("Result2: %d%n", result);
      nReturned++;        
    }
    assertEquals(nItems, nReturned);
  }
  
  /**
   * Sleeps for some random interval up to 3ms, then returns the input id.
   * 
   * @author Spence Green
   *
   */
  private static class DelayedIdentityFunction implements ThreadsafeProcessor<Integer,Integer> {

    // This class is not necessarily threadsafe
    //   http://docs.oracle.com/javase/1.4.2/docs/api/java/util/Random.html
    //   http://download.java.net/jdk7/archive/b123/docs/api/java/util/Random.html
    //
    // In Java 7, you can use ThreadLocalRandom:
    //   http://download.java.net/jdk7/archive/b123/docs/api/java/util/concurrent/ThreadLocalRandom.html
    //
    private final Random random = new Random();

    private static final int MAX_SLEEP_TIME = 3;

    @Override
    public Integer process(Integer input) {
      int sleepTime = nextSleepTime();
      try {
        Thread.sleep(sleepTime);
      } catch (InterruptedException e) {}
      return input;
    }

    private synchronized int nextSleepTime() {
      return random.nextInt(MAX_SLEEP_TIME);
    }

    @Override
    public ThreadsafeProcessor<Integer, Integer> newInstance() {
      return this;
    }
  }
}
