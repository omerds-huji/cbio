from Bio import SeqIO  # pip install biopython
from hmmlearn import hmm
from sklearn.metrics import classification_report

import argparse
import gzip
import numpy as np

# Maps for Nucleotides
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
# Maps for Labels
LABEL_MAP = {"N": 0, "C": 1}

# Define number of states and emissions in model
N_STATES = 8
N_EMISSIONS = 4

# 8-State Definition:
# States 0-3: A, C, G, T (Outside CpG)
# States 4-7: A, C, G, T (Inside CpG)


def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, "r") as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences


def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError(
            "Mismatch between sequence IDs and label IDs in the provided files."
        )

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]


def train_classifier(training_data):
    """
    Trains a classifier to identify CpG islands in DNA sequences.

    Parameters:
        training_data (list[tuple[str, str]]): Training data consisting of sequences and their labels.

    Returns:
        object: Your trained classifier model.
    """
    encoded_seqs = []
    encoded_labels = []

    # Convert seq to numerical seq
    # Convert label seq to 8 states numerical seq
    for seq, label in training_data:
        try:
            encoded_seq = np.array([NUC_MAP[b] for b in seq], dtype=int)
            is_cpg = np.array([LABEL_MAP[l] for l in label], dtype=int)
        except KeyError:  # seq or label are invalid, continue
            continue

        encoded_label = (4 * is_cpg) + encoded_seq

        encoded_seqs.append(encoded_seq)
        encoded_labels.append(encoded_label)

    # Calculate P_0
    start_counts = np.zeros(N_STATES)
    for label in encoded_labels:
        start_counts[label[0]] += 1
    start_prob = start_counts / (start_counts.sum() + 1e-10)

    # Calculate tranisition matrix
    transition_counts = np.ones((N_STATES, N_STATES))
    for label in encoded_labels:
        np.add.at(transition_counts, (label[:-1], label[1:]), 1)

    transition_mat = transition_counts / (
        transition_counts.sum(axis=1, keepdims=True) + 1e-10
    )

    # Calculate emission matrix
    emission_mat = np.zeros((N_STATES, N_EMISSIONS))
    for i in range(N_STATES):
        # P(A | A_in) = 1
        emission_mat[i, i % 4] = 1

    model = hmm.CategoricalHMM(n_components=N_STATES, init_params="")
    model.startprob_ = start_prob
    model.transmat_ = transition_mat
    model.emissionprob_ = emission_mat
    return model


def annotate_sequence(model: hmm.CategoricalHMM, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): Your trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """
    try:
        encoded_seq = np.array([NUC_MAP[b] for b in sequence], dtype=int)
    except KeyError:
        return ""

    encoded_seq = np.array(encoded_seq).reshape(-1, 1)
    _, state_seq = model.decode(encoded_seq, algorithm="map")

    # Recounstuct state seq
    predicted_states = []
    for state in state_seq:
        if state < 4:
            predicted_states.append("N")
        else:
            predicted_states.append("C")

    annotations = "".join(predicted_states)

    return annotations


def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, "wt") as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            if not annotation:
                continue
            gzipped_file.write(f">{seq_id}\n{annotation}\n")


def print_detailed_metrics(model, test_data):
    y_true = []
    y_pred = []

    for sequence, true_label in test_data:
        pred_label = annotate_sequence(model, sequence)
        if not pred_label:
            continue

        y_true.extend(list(true_label))
        y_pred.extend(list(pred_label))

    print(
        classification_report(
            y_true, y_pred, labels=["N", "C"], target_names=["Non-CpG", "CpG Island"]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict CpG islands in DNA sequences."
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        help="Path to the input FASTA file containing DNA sequences.",
    )
    # **Commented out, used for test validation**
    # parser.add_argument(
    #     "--label_path",
    #     type=str,
    #     help="Path to the input FASTA file containing DNA labels.",
    # )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output FASTA file for saving predictions.",
    )

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa"

    # Prepare training data and train model
    training_data = prepare_training_data(training_sequences_path, training_labels_path)
    classifier = train_classifier(training_data)

    # **Commented out, used for test validation**
    # test_data = prepare_training_data(args.fasta_path, args.label_path)
    # print_detailed_metrics(classifier, test_data)

    # Annotate sequences and save predictions
    annotate_fasta_file(classifier, args.fasta_path, args.output_path)
