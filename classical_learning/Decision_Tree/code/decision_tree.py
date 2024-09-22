import numpy as np
class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None


    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)


        if if_no_attribute(self.attribute_names):     #running out of attributes but you still have examples, now you should return the mode of examples
            tree = Tree(
                attribute_name="root",
                attribute_index=None,
                value=find_mode(targets),
                branches=[]
            )
            return tree

        if  np.array_equal(targets[1:], targets[:-1]): #3. all targets are the same or there's only one single value
            tree = Tree(
                attribute_name="root",
                attribute_index=None,
                value=(targets)[0],
                branches=[]
            )
            return tree
        else:
            attribute = find_best_attribute(features, self.attribute_names, targets)    #if all there are multiple attributes but they are garbage, we should stop and return the mode
            tree = Tree(
                attribute_name= attribute,   #I can set none to attribute_names that have been used. But you should never delete an attribute name
                attribute_index=self.attribute_names.index(attribute),
                value=find_mode(targets),
                branches=[]
            )
            if attribute != None:
                features_high , targets_high, features_low, targets_low = split_features_targets(features, targets, tree.attribute_index)   #TODO

                leaf_node = Tree(
                    attribute_name= attribute,
                    attribute_index=0,
                    value=find_mode(targets),
                    branches=[]
                )

                self.attribute_names[tree.attribute_index]=None         #remove the attribute name for the first tree!

                if len(targets_low) == 0:       #empty target lists
                    tree.branches.append(leaf_node)
                else:

                    tree.branches.append(self.fit(features_low, targets_low))       #the tree's branches will be [tree_for_low_value, tree_for_high_value]

                if len(targets_high) == 0:
                    tree.branches.append(leaf_node)
                else:
                    tree.branches.append(self.fit(features_high,targets_high))

                self.attribute_names[tree.attribute_index]=attribute         #recover the name so the other branch of the parent node will have the same set of inputs!!


            return tree


    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Return:
            predicted_target Nx1
        """
        self._check_input(features)
        predicted_target = np.array([])

        for index in range(len(features)):
            current_node = self.tree
            if current_node!= None:
                while len(current_node.branches):
                    if features[index,current_node.attribute_index]==1:  #the example's feature corresponding to the attribute index is High
                        current_node = current_node.branches[1]
                    else:                       #the example's feature corresponding to the attribute index is High
                        current_node = current_node.branches[0]
                predicted_target = np.append(predicted_target, current_node.value)
        return predicted_target


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def if_no_attribute(attribute_names):
    return len([False for attribute in attribute_names if attribute is not None ]) == 0

def find_mode(targets):
    _targets = (np.copy(targets)).tolist()
    return max(set(_targets), key = _targets.count)

def split_features_targets(features, targets, attribute_index):
    '''
    Split features table according to high and low values of an attribute_index.
    Also, split targets with the same row numbers.
    '''
    features_high = np.empty((0,len(features[0])),float)
    features_low = np.empty((0,len(features[0])),float)
    targets_high = np.array([])
    targets_low = np.array([])

    for index, val in np.ndenumerate(features[:,attribute_index]):
        if val == 1:
            features_high = np.vstack((features_high, features[index,:]))
            targets_high = np.append(targets_high, targets[index])
        else:
            features_low =np.vstack((features_low, features[index,:]))
            targets_low = np.append(targets_low, targets[index])
    return features_high , targets_high, features_low, targets_low

def find_best_attribute(features, attribute_names, targets):
    '''Return the best attribute name.
    Args:
        self.attribute_names (list that contains names or none)
        features: 2D np array
        targets: examples
    '''
    look_up_dict= {}
    for i in range(len(attribute_names)):
        if attribute_names[i]!=None:
            look_up_dict[attribute_names[i]]=information_gain(features, i, targets)

    sorted_look_up = sorted(look_up_dict.items(), key=lambda x: -x[1])
    if sorted_look_up[0][0]!= 0:        #if attributes are garbage, you'll get info gain zero
        return sorted_look_up[0][0]
    else:
        return None


def information_gain(features, attribute_index, targets):
    """
    Args:
        features (np.array): numpy array containing features for each example.  (table of features, each column is a feature's value: high or low)
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example. [No, yes, no yes...] Examples. Assume no is 0, yes is 1

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    p1 = np.sum(targets)/len(targets)
    p2 = 1.0-p1
    H_s = -p1*np.log2(p1)-p2*np.log2(p2) #use targets

    high_yes= 0
    high_No = 0
    low_yes = 0
    low_no = 0
    for index, feature_val in np.ndenumerate(features[:,attribute_index]):
        if feature_val==1:
            if targets[index] == 1:
                high_yes+=1
            else:
                high_No+=1
        else:
            if targets[index] == 1:
                low_yes+=1
            else:
                low_no+=1

    num_high = high_yes+high_No
    num_low = low_no+low_yes



    if num_high == 0:           #If unfortunately your features are all low, then this feature is useless. So info gain, the output should be zero
        return 0
    else:
        p_high_yes = high_yes/num_high
        p_high_no = high_No/num_high
        if p_high_yes==0 or p_high_no==0:
            H_s_high = 0
        else:
            H_s_high = num_high/len(targets)* (-p_high_yes*np.log2(p_high_yes)-p_high_no*np.log2(p_high_no))

    if num_low == 0:
        return 0
    else:
        p_low_yes = low_yes/num_low
        p_low_no = low_no/num_low
        if p_low_yes==0 or p_low_no==0:
            H_s_low = 0
        else:
            H_s_low = num_low/len(targets)*(-p_low_yes*np.log2(p_low_yes)- p_low_no*np.log2(p_low_no))

    H_sa =  H_s_high+H_s_low

    return H_s-H_sa


# if __name__ == '__main__':

#     attribute_names = ["Humidity", "wind"]
#     features = np.array([[1, 0],
#                          [1, 1],
#                          [1, 0],
#                          [1, 0],
#                          [0, 0],
#                          [0, 1],
#                          [0, 1],
#                          [1, 0],
#                          [0, 0],
#                          [0, 0],
#                          [0, 1],
#                          [1, 1],
#                          [0, 0],
#                          [1, 1]])    #only humidity
#     attribute_index = 0
#     targets = np.array([[0],
#                         [0],
#                         [1],
#                         [1],
#                         [1],
#                         [0],
#                         [1],
#                         [0],
#                         [1],
#                         [1],
#                         [1],
#                         [1],
#                         [1],
#                         [0]]).reshape(features.shape[0], )
#     decision_tree = DecisionTree(attribute_names=attribute_names)
#
#
#     features_high , targets_high, features_low, targets_low = split_features_targets(features, targets, attribute_index)
#     print (if_no_attribute([None, None]))
#     # decision_tree.tree = decision_tree.fit(features, targets)
#     # # decision_tree.visualize()
#     # # information_gain(features, attribute_index, targets)
#     # example_feature = np.array([[1,0],
#     #                             [0,1]])
#     # print (decision_tree.predict(example_feature))
