import unittest
from lavu.inventory_diagnostics import align_words,lexical_groups

class InventoryDiagnosticTests(unittest.TestCase):
    def test_new_ambiguity_distinguished_from_expanded_existing_classes(self):
        data={'entries':[['a','p1','u1'],['b','p1','u1'],['c','p2','u1'],['d','p3','u2']]}
        _,new,stats=lexical_groups(data)
        self.assertEqual(new,{'c'});self.assertEqual(stats['entries_with_expanded_competitor_sets'],3)
        self.assertEqual(stats['already_homophonous_entries_with_added_competitors'],2)
    def test_word_alignment_recovers_errors_with_declared_tie_policy(self):
        self.assertEqual(align_words('a b','a c d'),[('C','a','a'),('I',None,'c'),('S','b','d')])
        self.assertEqual(align_words('a b',''),[('D','a',None),('D','b',None)])
        self.assertEqual(align_words('','x'),[('I',None,'x')])
    def test_noncoarsening_class_export_is_rejected(self):
        with self.assertRaises(AssertionError):lexical_groups({'entries':[['a','p1','u1'],['b','p1','u2']]})

if __name__=='__main__':unittest.main()
