import java.awt.*;
import java.util.*;
import java.lang.*;
import java.io.*;


/**
 * Uses hidden Markov model trained on the Brown Corpus https://en.wikipedia.org/wiki/Brown_Corpus
 * in order to tag a word with their parts of speech
 *
 * @author garrettjohnston
 */


public class POSTagger {
    public static void main(String[] args) throws Exception {
        boolean test = false;
        boolean input = false;
        
        // Get training maps (observations/transitions) packed into list
        List<Map<String, Map<String, Double>>> training = train();
        
        // Test training data against user-selected testfiles
        if (test) 
            evaluateTagger(training); 

        // take user input to test tagger
        if (input) {
            while (true) {
                // Tag words from user input
                inputTagger(training); 
            }
        }
        
    }


    /**
     * Given training weight of state transitions and observations, return the most likely sequence of states(POS tags) based
     * on the training data
     * @param training List containing training data(observations, transitions)
     * @param line line of words to be tagged
     * @return Part of speech tags for given array
     */
    public static String viterbi(List<Map<String, Map<String, Double>>> training, String line) {
        Map<String, Map<String, Double>> observations = training.get(0);
        Map<String, Map<String, Double>> transitions = training.get(1);
        final double unseen = -100;  // Penalty for a word not seen by a certain tag in training

        String[] toTag = line.toLowerCase().split(" ");
        
        // Handle empty line
        if (toTag.length == 0) 
            return "";  

        // List of maps of tag -> previous tag to traverse backwards later
        List<Map<String,String>> backTrace = new ArrayList<>(toTag.length - 1);

        // Set and map of current tags, current tags -> score
        Set<String> currStates = new HashSet<>();
        Map<String, Double> currScores = new HashMap<>();
        currStates.add("#"); currScores.put("#", 0.0);  // Handle start case

        // Interrogate each observation(word) in line
        for (int i = 0; i < toTag.length; i++) {
            Map<String, String> bt = new HashMap<>();  // To put in backTrace
            Set<String> nextStates = new HashSet<>();  // States to visit on next iteration
            Map<String, Double> nextScores = new HashMap<>();  // Scores for next iteration
            
            for (String curr : currStates) {
                // Consider all states reachable from this one
                for (String next : transitions.get(curr).keySet()) {
                    nextStates.add(next);
                    // If next state observed in training, get score; otherwise, use unseen
                    double obScore = observations.get(next).getOrDefault(toTag[i], unseen);
                    
                    // Calculate score for transition to this state: current + transition score + observation score
                    double nextScore = currScores.get(curr) + transitions.get(curr).get(next) + obScore;
                    
                    // If there's a new tag or better score, put it in the map
                    if (!(nextScores.containsKey(next)) || (nextScore > nextScores.get(next))) { 
                        nextScores.put(next, nextScore);
                        bt.put(next, curr);  // curr is next's predecessor
                    }
                }
            }
            
            // update current states and scores, add best score to 
            currStates = nextStates;
            currScores = nextScores;
            backTrace.add(bt);
        }

        // Get tag with max score to traverse maps in backTrace
        String tracer = null;
        for (String tag : currScores.keySet()) {
            if (tracer == null || currScores.get(tag) > currScores.get(tracer)) 
                tracer = tag;
        }

        // Add all tags to a stack to get them out in the right order
        Stack<String> tagStack = new Stack<>();
        for (int i = backTrace.size() - 1; i >= 0; i--) {
            tagStack.push(tracer);
            tracer = backTrace.get(i).get(tracer);  // Traverse backwards through backTrace
        }

        StringBuilder allTags = new StringBuilder();
        while (tagStack.size() > 1) {
            allTags.append(tagStack.pop());
            allTags.append(" ");
        }
        allTags.append(tagStack.pop());

        return allTags.toString().toUpperCase();
    }


    /**
     * Trains HMM on user-selected text & tag files
     * @return List of observations(index 0), transitions(index 1) with log-probabilities
     * observations: state -> {observation -> weight} / tag -> {word -> weight}
     * transitions: state -> {state -> weight} / tag -> {tag -> weight} 
     */
    public static List<Map<String, Map<String, Double>>> train() throws IOException {
        // Select training files
        String obs = getPath(0);  // Training sentences file
        String tags = getPath(1);  // Training tags file
        Scanner obsScanner = new Scanner(new FileInputStream(obs));
        Scanner tagsScanner = new Scanner(new FileInputStream(tags));

        // Observation/transition maps to place frequencies in
        Map<String, Map<String, Double>> observations = new HashMap<>();  // tag -> (word -> freq)
        Map<String, Map<String, Double>> transitions = new HashMap<>();  // tag -> (other tag -> freq)
        transitions.put("#", new HashMap<>());  // Start character

        try {
            while (obsScanner.hasNextLine()) {
                // Assumes word & tag files have same number of lines
                String[] o = obsScanner.nextLine().toLowerCase().split(" "); // Words in a line
                String[] t = tagsScanner.nextLine().toLowerCase().split(" "); // Respective tags

                for (int i = 0; i < o.length; i++) {
                    String word = o[i];
                    String tag = t[i];
                    
                    /*
                     * Observations: current tag -> current word, freq
                     */ 
                    
                    if (!observations.containsKey(tag) { 
                        // Haven't seen this tag before
                        observations.put(tag, new HashMap<>());
                        observations.get(tag).put(word, 1.0);  // Place tag w/ word
                    } else if (!observations.get(tag).containsKey(word)) { 
                        // Seen current tag, but not current word
                        observations.get(tag).put(word, 1.0);  // Place current word in current tag's map
                    } else { 
                        // Seen this tag, and this word. Increment frequency
                        observations.get(tag).put(word, observations.get(tag).get(word) + 1);
                    }

                    /* 
                     * Transitions: current tag -> next tag, frequency
                     */
                    
                    if (i == 0) { 
                        // Deal with # -> first word
                        if (!transitions.get("#").containsKey(tag)) { 
                            // Haven't seen this tag as first in a line
                            transitions.get("#").put(tag, 1.0);
                        } else { 
                            // Seen tag as first in a line, increment frequency
                            transitions.get("#").put(tag, transitions.get("#").get(tag) + 1.0);
                        }
                    } else if (i == o.length - 1) { 
                        // Last word in line; no transition to i+1
                        if (!transitions.containsKey(tag)) 
                            transitions.put(tag, new HashMap<>());
                        continue;
                    }
                    
                    // Consider tag -> nextTag transition
                    String nextTag = t[i + 1];
                    if (!transitions.containsKey(tag)) { 
                        // Haven't seen this tag before
                        transitions.put(tag, new HashMap<>());
                        transitions.get(tag).put(nextTag, 1.0);
                    } else if (!transitions.get(tag).containsKey(nextTag)) {
                        // Seen this tag, but haven't seen next tag directly after this one
                        transitions.get(tag).put(nextTag, 1.0);
                    } else { 
                        // Seen tag and nextTag consecutively. Increment frequency of that sequence
                        transitions.get(tag).put(nextTag, transitions.get(tag).get(nextTag) + 1.0);
                    }
            }
        }
        } finally {
            obsScanner.close();
            tagsScanner.close();
        }

        // Convert to log-probability in place
        freqToWeight(observations);
        freqToWeight(transitions);

        // Pack up into list
        List<Map<String, Map<String, Double>>> weights = new ArrayList<>();
        weights.add(observations);
        weights.add(transitions);
        return weights;
    }


    /**
     * Helper for train()
     * Converts observation/transition map frequencies to log-probabilities in place
     * @param freq observation/transition map
     */
    public static void freqToWeight(Map<String, Map<String, Double>> freq) {
        for (String outer : freq.keySet()) {
            double tot = 0;  // Sum total occurrences
            for (String inner : freq.get(outer).keySet()) {
                tot += freq.get(outer).get(inner);
            }
            // Convert to log-probabilities
            for (String inner : freq.get(outer).keySet()) {
                freq.get(outer).put(inner, Math.log(freq.get(outer).get(inner)/tot));
            }
        }
    }


    /**
     * Print tags from a user-inputted line based on training data observations & transitions
     */
    public static void inputTagger(List<Map<String, Map<String, Double>>> training) {
        Scanner input = new Scanner(System.in);
        System.out.println("Type a sentence to tag");
        String userLine = input.nextLine();
        System.out.println(viterbi(training, userLine));
    }


    /**
     * Evaluate HMM tags based on training vs. true tags
     * Prints fraction of correctly tagged words
     * @param training observations & transitions packed in list
     */
    public static void evaluateTagger(List<Map<String, Map<String, Double>>> training) throws IOException {
        int correct = 0;  // Number of correctly tagged words
        int total = 0;
        String testObs = getPath(2);  // Test sentences file
        String testTags = getPath(3);  // Test tags file
        Scanner testObsScanner = new Scanner(new FileInputStream(testObs));
        Scanner testTagsScanner = new Scanner(new FileInputStream(testTags));

        try {
            int i = 0;
            while (testObsScanner.hasNextLine()) {
                String line = testObsScanner.nextLine();
                String tagged = viterbi(training, line); // model tags
                String realTags = testTagsScanner.nextLine(); // Actual tags

                String[] tags = tagged.split(" ");
                String[] real = realTags.split( " ");

                // Check correctness of this line
                int lineCorrect = 0;
                int lineTotal = 0;
                for (int j=0; j < tags.length; j++) {
                    lineTotal++;
                    if (tags[j].equals(real[j])) lineCorrect++;
                }

                // Display model tags vs. actual tags + fraction correct for the line
                System.out.println("Line " + i + ": " + line);
                System.out.println("Line " + i + " HMM tags: " + tagged);
                System.out.println("Line " + i + " true tags: " + realTags);
                System.out.println("Line " + i + " correct = " + lineCorrect + "/" + lineTotal);

                correct += lineCorrect;
                total += lineTotal;

                i++;
            }
        } finally {
            testObsScanner.close();
            testTagsScanner.close();
        }
        
        System.out.println(testObs + " correct: " + correct + "/" + total);
    }


    /**
     * User dialog for selecting a training/testing file
     * @return selected file path
     */
    public static String getPath(int type) {
        String title = "";
        switch(type) {
            case(0):
                title = "Select a file of sentences to train model";
                break;
            case(1):
                title = "Select a file of tags to train model";
                break;
            case(2):
                title = "Select a file of sentences to test model";
                break;
            case(3):
                title = "Select a file of tags to test model";
                break;
        }
        System.out.println(title);

        FileDialog dialog = new FileDialog((Frame)null, title);
        dialog.setMode(FileDialog.LOAD);
        dialog.setVisible(true);
        return (dialog.getDirectory() + dialog.getFile()).replace(System.getProperty("user.dir")+"/", "");
    }
}
