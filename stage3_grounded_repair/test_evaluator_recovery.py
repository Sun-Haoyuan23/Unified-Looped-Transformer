import unittest
from pathlib import Path
from recover_geometry_evaluation import compare_semantics


class RecoveryTests(unittest.TestCase):
    def test_semantic_changes_and_missing_scores_rejected(self):
        path=str(Path('anchor.png').resolve())
        row={'filename':path,'prompt':'fixed','correct':True,'reason':''}
        mapping={path:path}
        self.assertEqual(compare_semantics([row],[row],mapping)['n_anchors'],1)
        with self.assertRaises(ValueError):compare_semantics([row],[],mapping)
        with self.assertRaises(ValueError):compare_semantics([row],[dict(row,correct=False)],mapping)
        with self.assertRaises(ValueError):compare_semantics([row],[dict(row,prompt='wrong')],mapping)
        with self.assertRaises(ValueError):compare_semantics([row],[row,row],mapping)


if __name__=='__main__':unittest.main()
