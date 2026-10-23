"""Stdlib prompt/API validation; numerical torch/CFG checks run on the server."""

import unittest

import grounded_actions as actions


class GroundedPromptTests(unittest.TestCase):
    def test_all_target_relation_sides_and_subject_first(self):
        expected = {
            "left of": "a photo of a dog on the left, and a tie on the right.",
            "right of": "a photo of a dog on the right, and a tie on the left.",
            "above": "a photo of a dog on the top, and a tie on the bottom.",
            "below": "a photo of a dog on the bottom, and a tie on the top.",
        }
        for relation, prompt in expected.items():
            evidence = {"subject": "dog", "reference": "tie", "relation": relation}
            self.assertEqual(actions.concise_layout_prompt(evidence), prompt)
            self.assertEqual(evidence["relation"], relation)

    def test_counterfactual_swaps_sides_without_object_order_change(self):
        opposite = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}
        for relation, inverse in opposite.items():
            evidence = {"subject": "dog", "reference": "tie", "relation": relation}
            target = actions.concise_layout_prompt(dict(evidence, relation=inverse))
            self.assertEqual(actions.concise_layout_prompt(evidence, opposite=True), target)
            self.assertLess(target.index("dog"), target.index("tie"))

    def test_articles_and_multiword_objects(self):
        prompt = actions.concise_layout_prompt({"subject": "apple", "reference": "elephant", "relation": "left of"})
        self.assertEqual(prompt, "a photo of an apple on the left, and an elephant on the right.")
        prompt = actions.concise_layout_prompt({"subject": "hot dog", "reference": "orange", "relation": "above"})
        self.assertEqual(prompt, "a photo of a hot dog on the top, and an orange on the bottom.")
        prompt = actions.concise_layout_prompt({"subject": "unicorn", "reference": "umbrella", "relation": "below"})
        self.assertIn("a unicorn", prompt)
        self.assertIn("an umbrella", prompt)

    def test_object_names_and_relation_whitespace(self):
        prompt = actions.concise_layout_prompt({"subject": " hot   dog ", "reference": "sports ball", "relation": "  LEFT   OF "})
        self.assertEqual(prompt, "a photo of a hot dog on the left, and a sports ball on the right.")

    def test_invalid_evidence_fails_before_torch_import(self):
        invalid = [None, {}, {"subject": "", "reference": "tie", "relation": "left of"},
                   {"subject": "dog", "reference": "tie", "relation": "behind"}]
        for evidence in invalid:
            with self.assertRaises(ValueError):
                actions.concise_layout_prompt(evidence)
        with self.assertRaises(ValueError):
            actions.concise_layout_prompt({"subject": "dog", "reference": "tie", "relation": "above"}, opposite="yes")

    def test_field_dimensions_and_relation_validate_without_torch(self):
        for height, width in ((0, 3), (3, 0), (-1, 3), (True, 3), (1.5, 3)):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                actions.spatial_field("left of", height, width, "cpu")
        with self.assertRaisesRegex(ValueError, "Unsupported spatial relation"):
            actions.spatial_field("wrong axis", 3, 4, "cpu")
        self.assertNotIn("torch", vars(actions))

    def test_regional_strength_validates_before_runtime_access(self):
        for strength in (float("nan"), float("inf"), -.1, 1.1, True):
            with self.assertRaisesRegex(ValueError, "strength"):
                actions.regional_velocity(None, None, None, None, {}, strength=strength)

    def test_teacher_note_discloses_original_prompt_scoring(self):
        self.assertIn("Offline regional velocity teacher", actions.REGIONAL_TEACHER_NOTE)
        self.assertIn("not the core shared-loop method", actions.REGIONAL_TEACHER_NOTE)
        self.assertIn("original prompt", actions.REGIONAL_TEACHER_NOTE)


if __name__ == "__main__":
    unittest.main()
