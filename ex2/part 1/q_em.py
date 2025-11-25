import numpy as np


# =====================================================
#                 FASTA SEQUENCE READER
# =====================================================
def read_fasta_sequences(path):
    """Reads FASTA sequences into a list of strings."""
    seqs = []
    with open(path) as f:
        current = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append("".join(current))
    return seqs


# =====================================================
#                   HMM_EM CLASS
# =====================================================
class HMM_EM:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations

        N = len(states)
        M = len(observations)

        # -------------------------------------------------------
        # INITIALIZATIONS (kept exactly as you requested)
        # -------------------------------------------------------
        # Initial state probabilities
        self.initial = np.array([0.5, 0.5])

        # Transition matrix
        self.transition = np.random.rand(N, N)
        self.transition = self.transition / self.transition.sum(axis=1, keepdims=True)

        # Emission matrix
        self.emission = np.random.rand(N, M)
        self.emission = self.emission / self.emission.sum(axis=1, keepdims=True)

        # Print initialized parameters (for debugging)
        print("Initial state distribution:\n", self.initial)
        print("Transition matrix:\n", self.transition)
        print("Emission matrix:\n", self.emission)

        # Observation → index lookup
        self.obs_index = {o: i for i, o in enumerate(observations)}

    # =====================================================
    #                   FORWARD ALGORITHM
    # =====================================================
    def forward(self, obs_seq):
        """
        Compute forward probabilities α_t(i).

        Returns
        -------
        alpha : np.ndarray of shape (T, N)
        """
        T = len(obs_seq)
        N = len(self.states)
        alpha = np.zeros((T, N))
        # Convert observations to indices
        # Use lecture's terminology - X for the seq of observations
        X = [self.obs_index[o] for o in obs_seq]

        # Calculate initial states
        alpha[0] = self.initial * self.emission[:, X[0]]

        for i in range(1, T):
            for s in range(N):
                alpha[i, s] = self.emission[s, X[i]] * np.sum(
                    alpha[i - 1] * self.transition[:, s]
                )
        return alpha

    # =====================================================
    #                   BACKWARD ALGORITHM
    # =====================================================
    def backward(self, obs_seq):
        """
        Compute backward probabilities β_t(i).

        Returns
        -------
        beta : np.ndarray of shape (T, N)
        """
        T = len(obs_seq)
        N = len(self.states)
        beta = np.zeros((T, N))
        # Convert observations to indices
        # Use lecture's terminology - X for the seq of observations
        X = [self.obs_index[o] for o in obs_seq]

        # Calculate final states
        beta[T - 1] = np.ones(N)
        for i in range(T - 2, -1, -1):
            for s in range(N):
                beta[i, s] = np.sum(
                    self.transition[s, :] * self.emission[:, X[i + 1]] * beta[i + 1]
                )
        return beta

    # =====================================================
    #                   BAUM–WELCH / EM
    # =====================================================
    def baum_welch(self, seqs, max_iter=150):
        """
        Train the HMM using Baum–Welch EM.

        Returns
        -------
        transition : np.ndarray
        emission : np.ndarray
        log_likelihoods : list
            Log-likelihood at each iteration
        """
        N = len(self.states)
        M = len(self.observations)
        tol = 1e-4
        log_likelihoods = []
        last_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # Expected number of transitions from state k to l
            expected_transitions = np.zeros((N, N))
            # Expected number of times in state k emitting symbol x
            expected_emissions = np.zeros((N, M))

            total_log_likelihood = 0

            # E-STEP: Loop over every sequence X^j
            for seq in seqs:
                T = len(seq)
                obs_indices = [self.obs_index[o] for o in seq]

                # Calculate forward and backward
                alpha = self.forward(seq)
                beta = self.backward(seq)

                # Calculate P(X^j) - Likelihood of this sequence
                seq_likelihood = np.sum(alpha[T - 1])
                total_log_likelihood += np.log(seq_likelihood)

                # gamma[t, k] = probability of being in state k at time t
                gamma = (alpha * beta) / seq_likelihood
                # Counts for emissions (N_k,x)
                for t in range(T):
                    expected_emissions[:, obs_indices[t]] += gamma[t, :]

                # xi[t, k, l] = prob of transition k->l at time t
                xi = np.zeros((T - 1, N, N))
                for t in range(T - 1):
                    numerator = (
                        alpha[t, :, np.newaxis]
                        * self.transition
                        * self.emission[:, obs_indices[t + 1]]
                        * beta[t + 1, :]
                    )
                    xi[t] = numerator / seq_likelihood
                # Counts for Transitions (N_k,l)
                expected_transitions += np.sum(xi, axis=0)

            # M-STEP: Update Parameters
            # Update transition matrix
            self.transition = expected_transitions / (
                expected_transitions.sum(axis=1, keepdims=True) + 1e-12
            )
            # Update emission matrix
            self.emission = expected_emissions / (
                expected_emissions.sum(axis=1, keepdims=True) + 1e-12
            )
            # Update log likelihood array
            log_likelihoods.append(total_log_likelihood)
            print(
                f"Iteration {iteration + 1}: Log-Likelihood = {total_log_likelihood:.4f}"
            )

            # Check Convergence
            delta = total_log_likelihood - last_log_likelihood
            if abs(delta) < tol and iteration > 0:
                print(
                    f"Converged at iteration {iteration}, Log-Likelihood: {total_log_likelihood:.4f}"
                )
                break

            last_log_likelihood = total_log_likelihood

        return self.transition, self.emission, log_likelihoods


# =============================================================
#                   MAIN PROGRAM (Student Entry Point)
# =============================================================
if __name__ == "__main__":
    sequences = read_fasta_sequences("data/q_coins.seq.fa")

    hmm = HMM_EM(states=["Fair", "Loaded"], observations=["H", "T"])

    try:
        transition, emission, _ = hmm.baum_welch(sequences, max_iter=150)

        print("\nLearned Transition Matrix:")
        print(transition)

        print("\nLearned Emission Matrix:")
        print(emission)

    except NotImplementedError as e:
        print("Missing implementation:", e)
