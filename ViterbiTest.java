import java.util.*;
import java.lang.*;

/**
 *
 * Test class for POSTagger.viterbi
 * @author garrettjohnston
 */

public class ViterbiTest {
    public static void main(String[] args) {
        // Ice cream example from HMM notes
        Map<String, Map<String, Double>> t1Observations = new HashMap<>();
        Map<String, Map<String, Double>> t1Transitions = new HashMap<>();

        // Start transitions
        t1Transitions.put("#", new HashMap<>());
        t1Transitions.get("#").put("cold", Math.log((double) 5/10));
        t1Transitions.get("#").put("hot", Math.log((double) 5/10));
        // Hot transitions
        t1Transitions.put("hot", new HashMap<>());
        t1Transitions.get("hot").put("hot", Math.log((double) 7/10));
        t1Transitions.get("hot").put("cold", Math.log((double) 3/10));
        // Hot observations
        t1Observations.put("hot", new HashMap<>());
        t1Observations.get("hot").put("1", Math.log((double) 2/10));
        t1Observations.get("hot").put("2", Math.log((double) 3/10));
        t1Observations.get("hot").put("3", Math.log((double) 5/10));
        // Cold transitions
        t1Transitions.put("cold", new HashMap<>());
        t1Transitions.get("cold").put("cold", Math.log((double) 7/10));
        t1Transitions.get("cold").put("hot", Math.log((double) 3/10));
        // Cold observations
        t1Observations.put("cold", new HashMap<>());
        t1Observations.get("cold").put("1", Math.log((double) 7/10));
        t1Observations.get("cold").put("2", Math.log((double) 2/10));
        t1Observations.get("cold").put("3", Math.log((double) 1/10));
        List<Map<String, Map<String, Double>>> training1 = new ArrayList<>();
        training1.add(t1Observations); training1.add(t1Transitions);

        final String t1Sequence1 = "2 3 2 1";
        final String t1Sequence2 = "1 1 1 1";
        final String t1Sequence3 = "1 2 1 2";
        // Should print hot hot hot cold, as in notes
        System.out.println("Test 1a: expect hot hot hot cold");
        System.out.println(POSTagger.viterbi(training1, t1Sequence1));
        // Expect all cold - lowest amount every day
        System.out.println("Test 1b: expect cold cold cold cold");
        System.out.println(POSTagger.viterbi(training1, t1Sequence2));
        // Expect all cold again - high likelihood of staying same temperature
        System.out.println("Expect cold cold cold cold");
        System.out.println(POSTagger.viterbi(training1, t1Sequence3));


        // Example from section 5/19
        Map<String, Map<String, Double>> t2Observations = new HashMap<>();
        Map<String, Map<String, Double>> t2Transitions = new HashMap<>();
        // Start transitions
        t2Transitions.put("#", new HashMap<>());
        t2Transitions.get("#").put("n", Math.log((double) 5/7));
        t2Transitions.get("#").put("np", Math.log((double) 2/7));
        // cnj transitions
        t2Transitions.put("cnj", new HashMap<>());
        t2Transitions.get("cnj").put("n", Math.log((double) 1/3));
        t2Transitions.get("cnj").put("np", Math.log((double) 1/3));
        t2Transitions.get("cnj").put("v", Math.log((double) 1/3));
        // n transitions
        t2Transitions.put("n", new HashMap<>());
        t2Transitions.get("n").put("cnj", Math.log((double) 2/8));
        t2Transitions.get("n").put("v", Math.log((double) 6/8));
        // np transitions
        t2Transitions.put("np", new HashMap<>());
        t2Transitions.get("np").put("v", Math.log((double) 2/2));
        // v transitions
        t2Transitions.put("v", new HashMap<>());
        t2Transitions.get("v").put("cnj", Math.log((double) 1/9));
        t2Transitions.get("v").put("n", Math.log((double) 6/9));
        t2Transitions.get("v").put("np", Math.log((double) 2/9));
        // cnj observations
        t2Observations.put("cnj", new HashMap<>());
        t2Observations.get("cnj").put("and", Math.log((double) 3/3));
        // n observations
        t2Observations.put("n", new HashMap<>());
        t2Observations.get("n").put("cat", Math.log((double) 5/12));
        t2Observations.get("n").put("dog", Math.log((double) 5/12));
        t2Observations.get("n").put("watch", Math.log((double) 2/12));
        // np observations
        t2Observations.put("np", new HashMap<>());
        t2Observations.get("np").put("chase", Math.log((double) 5/5));
        // v observations
        t2Observations.put("v", new HashMap<>());
        t2Observations.get("v").put("chase", Math.log((double) 2/9));
        t2Observations.get("v").put("get", Math.log((double) 1/9));
        t2Observations.get("v").put("watch", Math.log((double) 6/9));

        List<Map<String, Map<String, Double>>> training2 = new ArrayList<>();
        training2.add(t2Observations); training2.add(t2Transitions);

        // Expect NP V N V N as given in section
        final String t2Sequence1 = "chase watch dog chase watch";
        // Expect V CNJ N V NP, but watch is seen much more often as a noun in training.
        // "and" is never seen in training, so result N V N V NP makes sense.
        final String t2Sequence2 = "watch the dog chase chase";
        System.out.println("Test 2a: expect NP V N V N");
        System.out.println(POSTagger.viterbi(training2, t2Sequence1));
        System.out.println("Test 2b: expect V CNJ N V NP");
        System.out.println(POSTagger.viterbi(training2, t2Sequence2));
    }

}
