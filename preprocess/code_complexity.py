from nltk.tokenize import word_tokenize
import math
from dataset_creation.xmlhelpers import ast_to_xml_tree
from collections import Counter



def cyclomatic_complexity(source_code):
    """
    Calculate the McCabe Cyclomatic Complexity for Java snippet manually without a flow control graph

    :param source_code
    :return: cc
    """
    cc = 1

    control_flow_statements = ["if", "else", "else if", "while", "for", "and"]

    tokens = word_tokenize(source_code)

    for w in control_flow_statements:
        cc = cc + tokens.count(w)

    return cc

def get_number_of_names(ast):
    """
    Calculate the number of variables,methods and objects of a snippet

    :param ast
    :return:
    """

    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("SimpleName"):
        counter += 1

    return counter

def get_number_of_lines_changed(code_changes):
    number_of_lines = len(code_changes.splitlines())
    return number_of_lines
    
def contains_type_of_block(code_changes):
    type_of_blocks = ["for", "while", "do", "switch", "if", "try"]
    
    for_flag = 0
    while_flag = 0
    do_flag = 0
    switch_flag = 0
    if_flag = 0
    try_flag = 0
    
    if type_of_blocks[0] in code_changes:
        for_flag = 1
    elif type_of_blocks[1] in code_changes:
        while_flag = 1
    elif type_of_blocks[2] in code_changes:
        do_flag = 1
    elif type_of_blocks[3] in code_changes:
        switch_flag = 1
    elif type_of_blocks[4] in code_changes:
        if_flag = 1
    elif type_of_blocks[5] in code_changes:
        try_flag = 1
    
    return for_flag,while_flag,do_flag,switch_flag,if_flag,try_flag

def get_unique_operators_count(code_changes):

    operators = ['+','-','*','/','%','++','--','=','+=','-=','*=','/=','%=','&=','|=','^=','>>=','<<=',\
        '==','!=','>','<','>=','<=','&&','||','!','{','}','(',')']
    
    unique_operators_counter = 0
    
    for operator in operators:
        if Counter(code_changes)[operator] > 0 :
            unique_operators_counter += 1
    
    return unique_operators_counter

def get_number_of_method_invocations(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("MethodInvocation"):
        counter += 1

    return counter

def get_number_of_blocks(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("Block"):
        counter += 1

    return counter

def get_number_of_if_statements(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("IfStatement"):
        counter += 1

    return counter

def get_number_of_variable_statements(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("VariableDeclarationStatement"):
        counter += 1

    return counter

def get_number_of_expression_statements(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ExpressionStatement"):
        counter += 1

    return counter

def get_number_of_infix_expression(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("InfixExpression"):
        counter += 1

    return counter

def get_number_of_prefix_expression(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("PrefixExpression"):
        counter += 1

    return counter

def get_number_of_switch_statements(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("SwitchStatement"):
        counter += 1

    return counter

def get_number_of_conditional_expression(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ConditionalExpression"):
        counter += 1

    return counter

def get_number_of_parenthesized_expression(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ParenthesizedExpression"):
        counter += 1

    return counter

def get_number_of_brake_statement(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("BreakStatement"):
        counter += 1

    return counter

def get_number_of_class_instance_creations(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ClassInstanceCreation"):
        counter += 1

    return counter

def get_number_of_while_statement(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("WhileStatement"):
        counter += 1

    return counter

def get_number_of_for_statement(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ForStatement"):
        counter += 1

    
    for n in xml_tree.iter("EnhancedForStatement"):
        counter += 1
    
    return counter

def get_number_of_try_statement(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("TryStatement"):
        counter += 1

    return counter

def get_number_of_return_statement(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("ReturnStatement"):
        counter += 1

    return counter

def get_number_of_single_variable_delcaration(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("SingleVariableDeclaration"):
        counter += 1

    return counter

def get_number_of_qualified_name(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    counter = 0

    for n in xml_tree.iter("QualifiedName"):
        counter += 1

    return counter

def get_number_of_children(ast):
    
    xml_tree = ast_to_xml_tree(ast)
    result = len(xml_tree.getchildren())
    
    return result