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
        # -------------------------------------------------------
        # TODO: Implement forward algorithm
        # -------------------------------------------------------
        raise NotImplementedError("Forward algorithm not implemented.")

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
        # -------------------------------------------------------
        # TODO: Implement backward algorithm
        # -------------------------------------------------------
        raise NotImplementedError("Backward algorithm not implemented.")

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
        """
        # -------------------------------------------------------
        # TODO: Implement Baum–Welch (EM)
        # -------------------------------------------------------
        raise NotImplementedError("Baum–Welch EM not implemented.")


# =============================================================
#                   MAIN PROGRAM (Student Entry Point)
# =============================================================
if __name__ == "__main__":
    sequences = read_fasta_sequences("data/q_coins.seq.fa")

    hmm = HMM_EM(
        states=["Fair", "Loaded"],
        observations=["H", "T"]
    )

    try:
        transition, emission = hmm.baum_welch(sequences, max_iter=150)

        print("\nLearned Transition Matrix:")
        print(transition)

        print("\nLearned Emission Matrix:")
        print(emission)

    except NotImplementedError as e:
        print("Missing implementation:", e)
