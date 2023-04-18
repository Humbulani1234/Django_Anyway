# with open('test.csv', 'w') as f:
#     g = str(X_test.iloc[0].tolist())
#     f.write(g)

#import pandas as pd

# def getDuplicateColumns(df):

#     # Create an empty set
#     duplicateColumnNames = set()

#     # Iterate through all the columns
#     # of dataframe
#     for x in range(df.shape[1]):

#         # Take column at xth index.
#         col = df.iloc[:, x]

#         # Iterate through all the columns in
#         # DataFrame from (x + 1)th index to
#         # last index
#         for y in range(x + 1, df.shape[1]):

#             # Take column at yth index.
#             otherCol = df.iloc[:, y]

#             # Check if two columns at x & y
#             # index are equal or not,
#             # if equal then adding
#             # to the set
#             if col.equals(otherCol):
#                 duplicateColumnNames.add(df.columns.values[y])

#     # Return list of unique column names
#     # whose contents are duplicates.
#     return list(duplicateColumnNames)


#     # Get list of duplicate columns
#     #duplicateColNames = getDuplicateColumns(df)

#     #for column in duplicateColNames:
#         #print('Column Name : ', column)

# b = getDuplicateColumns(X_test)
# print(b)