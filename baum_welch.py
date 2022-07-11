import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
import dataframe_image as dfi


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
        self.transitions = [0,1,2] # S

        self.a_matrix = np.zeros((len(self.transitions), len(self.transitions)))
        self.b_matrix = np.zeros((len(self.transitions), len(self.states)))
        #! remove next line later
        self.sequence_output()
        self.alpha = np.zeros((len(self.transitions), len(self.output)))
        self.beta = np.zeros((len(self.transitions), len(self.output)))

        self.fill_a_matrix()
        self.fill_b_matrix()
        self.sequence_output()

    def fill_a_matrix(self):
        # Transition Probabilities
        self.a_matrix = np.array([
            [0, 0.5, 0.5],
            [0.25, 0.5, 0.25],
            [1/3, 0, 2/3]])

    def fill_b_matrix(self):
        # Emission Probabilities
        self.b_matrix = np.array([
            [1, 0, 0, 0, 0],
            [0, 1/8, 1/8, 1/4, 1/2],
            [0, 1/2, 1/4, 1/8, 1/8]])
        self.b_matrix = pd.DataFrame(self.b_matrix, columns=self.states)

    def sequence_output(self):
        self.output = ",CGG,"

    def forward(self):
        self.alpha = np.zeros((len(self.transitions), len(self.output)))
        self.alpha[0][0] = 1
        # start at second row (first is initial), start at index 1 because the first row was skipped
        for row_index, row in enumerate(self.alpha.T[1:], 1):
            for cell_index, cell in enumerate(row):
                #* Idea: sum(Last alpha row * fitting A Row) * cell from B
                last_alpha_row = np.copy(self.alpha.T[row_index-1])
                a_matrix_row = np.copy(self.a_matrix.T[cell_index])
                matrix_multiply_result_sum = np.dot(last_alpha_row, a_matrix_row)

                current_nucleo = self.output[row_index]
                nucleo_probability = self.b_matrix[current_nucleo][cell_index]
                # 4 ndigits
                cell_result = round(matrix_multiply_result_sum * nucleo_probability, 8)
                self.alpha.T[row_index][cell_index] = cell_result

    def backward(self):
        self.beta = np.zeros((len(self.transitions), len(self.output)))
        self.beta[0][-1] = 1
        # start at second row (first is initial), start at index 2 because the last row was skipped and negative starts at -1,
        # [1::-1] means backward loop to second row
        for col_index, col in enumerate(self.beta.T[-2::-1], 2):
            for row_index, val in enumerate(col):

                a_matrix_col = np.copy(self.a_matrix[row_index])

                next_nucleo = self.output[-col_index+1]
                b_matrix_col = np.copy(self.b_matrix[next_nucleo].to_numpy())
                next_beta_column = np.copy(self.beta.T[-col_index+1])
                arrays = [a_matrix_col, next_beta_column, b_matrix_col]
                matrix_multiply_result_sum = sum(
                    np.prod(np.vstack(arrays), axis=0))
                # 4 ndigits
                cell_result = round(matrix_multiply_result_sum, 8)
                self.beta.T[-col_index][row_index] = cell_result

    def create_image(self, df: pd.DataFrame):
        df = pd.DataFrame(df, columns=[x for x in self.output])

        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
        tabla = table(ax, df, loc='upper right', colWidths=[
                    0.17]*len(df.columns))  # where df is your data frame
        tabla.auto_set_font_size(False)  # Activate set fontsize manually
        tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
        tabla.scale(1.2, 1.2)  # change size table
        plt.savefig('table.png', transparent=True)

    def baum_welch(self):
        max_loops = 100
        loop = 1
        while loop < max_loops:
            last_alpha_result = self.alpha[0][-1]
            self.forward()
            new_alpha_result = self.alpha[0][-1]
            if last_alpha_result == new_alpha_result:
                print(f"Calculation stopped - stable Possibility : {round(new_alpha_result,8)*100}% after {loop} loops!\n")
                return
            self.backward()

            def recalculate_a():
                for row_index, row in enumerate(self.a_matrix):
                    for column_index, value in enumerate(row):
                        nucleos = self.output[1:]
                        empty_b_list = []
                        for n in nucleos:
                            empty_b_list.append(self.b_matrix[n][column_index])
                        b_matrix_values = np.array(empty_b_list)
                        arrays = [b_matrix_values, np.copy(self.alpha[row_index][:-1]), np.copy(self.beta[column_index][1:])]

                        counter = sum(np.prod(np.vstack(arrays), axis=0) * self.a_matrix[row_index][column_index])
                        denominator = np.dot(
                            np.copy(self.alpha[row_index][1:]), np.copy(self.beta[row_index][1:]))

                        if (counter != 0) and (denominator != 0):
                            result = counter / denominator
                        else:
                            result = 0

                        self.a_matrix[row_index][column_index] = round(result, 8)

            def recalculate_b():
                for row_index, row in self.b_matrix.iloc[1:].iterrows():
                    for column_index, value in enumerate(row):
                        # find all indexes from output, where current b matrix column name appears
                        all_nucleo_indexes = [i for i, ltr in enumerate(
                            self.output) if ltr == self.b_matrix.columns[column_index]]

                        counter = np.dot(
                            self.alpha[row_index][all_nucleo_indexes], self.beta[row_index][all_nucleo_indexes])
                        denominator = np.dot(self.alpha[row_index], self.beta[row_index])
                        if counter != 0 or denominator != 0:
                            result = counter / denominator
                        else:
                            result = 0
                        self.b_matrix[self.b_matrix.columns[column_index]].iloc[row_index] = round(result,8)

        recalculate_a()
        recalculate_b()

        loop += 1

    def check(self) -> bool:
        # Check if forward and backword worked
        return hmm.beta[0][0] == round(np.dot(hmm.beta.T[1], hmm.alpha.T[1]), 8)

    def __repr__(self) -> str:
        a_df = pd.DataFrame(
            self.a_matrix, columns=self.transitions)
        b_df = pd.DataFrame(self.b_matrix, columns=self.states)
        alpha_df = pd.DataFrame(self.alpha, columns=[x for x in self.output])
        beta_df = pd.DataFrame(self.beta, columns=[x for x in self.output])

        rep = "Transition Probabilities - A-Matrix\n" + \
            f"{a_df}\n\n" + "Emission Probabilities - B-Matrix\n" + \
            f"{b_df}" + "\n\nAlpha\n" + f"{alpha_df}" + \
            f"\nBeta\n" + f"{beta_df}"
        return rep


hmm = HiddenMarkovModel()
print(hmm)

print("\n\n =========== BAUM-WELCH START ==============\n")
hmm.baum_welch()
print(hmm)


# hmm.create_image(hmm.alpha)
