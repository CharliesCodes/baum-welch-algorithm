import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import table

# States
# # V = data['Visible'].values

""" Vorgaben
    HMM als class

    -------------
    def set_states()
    def a_matrix()
    def b_matrix()
    sef sequenz_output()
    ------------->         INPUT

    def forward()
        -> svg in jedem Schritt erzeugen
        aufruf: next? -> nächster forward Schritt

    def backward()
        siehe next

    baum-welch()
        ~ so lala
"""


"""SVG VORLAGE
    <svg>
    <symbol id="player">
        <circle cx="225" cy="25" r="20" stroke="black" stroke-width="2" fill="red"/>
    </symbol>
    <path id="myarrow" d="M5,15 H25 V5 L45,25 L25,45 V35 H5 z"  fill="yellow" stroke="#ff6666" stroke-width="5" stroke-linejoin="round" />
    <use href="#myarrow" x="10" fill="red" />

    <circle id="myCircle" cx="50" cy="150" r="40" stroke="blue"/>
    <use href="#myCircle" x="50" fill="blue"/>
    <use href="#myCircle" x="100" fill="red" stroke="red"/>
    <use href="#player" />
    </svg>

"""



class HiddenMarkovModel:

    def __init__(self):
        self.states = [",", "A", "T", "G", "C"]  # V
        self.transitions = [1,2,3] # S

        self.a_matrix = np.zeros((len(self.transitions), len(self.transitions)))
        self.b_matrix = np.zeros((len(self.transitions), len(self.states)))
        #! remove next line later
        self.sequence_output()
        self.alpha = np.zeros((len(self.transitions), len(self.output)))
        self.beta = np.zeros((len(self.transitions), len(self.output)))

    def fill_a_matrix(self):
        # Transition Probabilities
        self.a_matrix = np.array([
            [0, 0.5, 0.5],
            [0, 0.25, 0.75],
            [0.5, 0.25, 0.25]])

    def fill_b_matrix(self):
        # Emission Probabilities
        self.b_matrix = np.array([
            [1, 0, 0, 0, 0],
            [0, 0.25, 0.5, 0.125, 0.125],
            [0, 0.5, 0.25, 0.125, 0.125]])
        self.b_matrix = pd.DataFrame(self.b_matrix, columns=self.states)

    def sequence_output(self):
        self.output = ",ATTGA,"

    def forward(self):
        self.alpha[0][0] = 1
        # start at second row (first is initial), start at index 1 because the first row was skipped
        for row_index, row in enumerate(self.alpha.T[1:], 1):
            for cell_index, cell in enumerate(row):
                #* Idea: sum(Last alpha row * fitting A Row) * cell from B
                last_alpha_row = self.alpha.T[row_index-1]
                a_matrix_row = self.a_matrix.T[cell_index]
                matrix_multiply_result_sum = np.dot(last_alpha_row, a_matrix_row)

                current_nucleo = self.output[row_index]
                nucleo_probability = self.b_matrix[current_nucleo][cell_index]
                # 4 ndigits
                cell_result = round(matrix_multiply_result_sum * nucleo_probability, 3)
                self.alpha.T[row_index][cell_index] = cell_result
                alpha_df = pd.DataFrame(self.alpha, columns=[x for x in self.output])
                print("\n\nAlpha\n" + f"{alpha_df}")

    def backward(self):
        self.beta = np.copy(self.alpha)
        print(self.beta)
        # start at second row (first is initial), start at index 2 because the last row was skipped and negative starts at -1,
        # [1::-1] means backward loop to second row
        for row_index, row in enumerate(self.beta.T[-2:0:-1], 2):
            print(-row_index, row)
            print(self.beta.T)
            for cell_index, cell in enumerate(row[::-1]):
                print(cell, -cell_index-1)
                # 4 ndigits
                # cell_result = round(
                #     matrix_multiply_result_sum * nucleo_probability, 3)
                # self.beta.T[-row_index][-cell_index-1] = cell_result

    def __repr__(self) -> str:
        a_df = pd.DataFrame(
            self.a_matrix, columns=self.transitions)
        b_df = pd.DataFrame(self.b_matrix, columns=self.states)
        alpha_df = pd.DataFrame(self.alpha, columns=[x for x in self.output])
        rep = "Transition Probabilities - A-Matrix\n" + \
            f"{a_df}\n\n" + "Emission Probabilities - B-Matrix\n" + \
            f"{b_df}" + "\n\nAlpha\n" + f"{alpha_df}"
        return rep



hmm = HiddenMarkovModel()
hmm.fill_a_matrix()
hmm.fill_b_matrix()
hmm.sequence_output()
hmm.forward()
# hmm.backward()



# ax = plt.subplot(111, frame_on=False)  # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis

# test = pd.DataFrame(hmm.alpha, columns=[x for x in hmm.output])
# table(ax, test)  # where df is your data frame

# plt.savefig('mytable.svg')


# print(hmm)