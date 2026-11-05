"""Integrity checks for the paired role-reversal lesson."""
import unittest

from forge_core.neural.role_counterfactual_v1 import (
    FINAL_OBJECTS, FINAL_PEOPLE, SKILLS, TRAIN_OBJECTS, TRAIN_PEOPLE,
    TRAIN_STORY_TEMPLATES, make_final, make_train, shuffle_roles,
)


class TestRoleCounterfactualV1(unittest.TestCase):
    def test_splits_are_disjoint_and_reproducible(self):
        self.assertFalse(set(TRAIN_PEOPLE) & set(FINAL_PEOPLE))
        self.assertFalse(set(TRAIN_OBJECTS) & set(FINAL_OBJECTS))
        training = make_train(400, seed=110)
        final = make_final(50, seed=101)
        self.assertEqual(training, make_train(400, seed=110))
        self.assertEqual(final, make_final(50, seed=101))
        self.assertFalse({row.prompt for row in training} &
                         {row.prompt for row in final})
        self.assertEqual(len(final), 200)
        self.assertEqual(sum(row.stratum == "unseen_verb" for row in final), 100)

    def test_every_training_episode_is_a_four_way_counterfactual(self):
        rows = make_train(160, seed=211)
        episodes = {}
        for row in rows:
            episodes.setdefault(row.episode, []).append(row)
        self.assertEqual(len(episodes), 40)
        for bundle in episodes.values():
            self.assertEqual(len(bundle), 4)
            self.assertEqual({row.skill for row in bundle}, set(SKILLS))
            self.assertEqual({row.answer for row in bundle if row.skill == "giver"},
                             {row.answer for row in bundle if row.skill == "recipient"})
            stories = [row.prompt.split("\nQuestion:", 1)[0] for row in bundle]
            self.assertEqual(len(set(stories)), 2)
            self.assertEqual(stories[0], stories[1])
            # Each direction has a different answer for each role.
            giver = [row.answer for row in bundle if row.skill == "giver"]
            recipient = [row.answer for row in bundle if row.skill == "recipient"]
            self.assertNotEqual(giver[0], giver[1])
            self.assertNotEqual(recipient[0], recipient[1])

    def test_final_templates_and_heldout_verbs_are_isolated(self):
        train = "\n".join(row.prompt for row in make_train(800, seed=17)).casefold()
        final = make_final(50, seed=18)
        self.assertTrue(all(template.split()[1].lower() in train
                            for template in TRAIN_STORY_TEMPLATES))
        self.assertNotIn("transferred", train)
        self.assertNotIn("delivered", train)
        novel = [row for row in final if row.stratum == "unseen_verb"]
        self.assertEqual(len(novel), 100)
        self.assertTrue(any("transferred" in row.prompt.lower() for row in novel))
        self.assertTrue(any("delivered" in row.prompt.lower() for row in novel))

    def test_shuffled_control_preserves_prompts_and_role_balance(self):
        rows = make_train(160, seed=91)
        null = shuffle_roles(rows, seed=777)
        self.assertEqual([row.prompt for row in rows], [row.prompt for row in null])
        self.assertNotEqual([row.answer for row in rows], [row.answer for row in null])
        for skill in SKILLS:
            self.assertEqual(
                sorted(row.answer for row in rows if row.skill == skill),
                sorted(row.answer for row in null if row.skill == skill),
            )

    def test_bounds_reject_invalid_size(self):
        with self.assertRaises(ValueError):
            make_train(42)
        with self.assertRaises(ValueError):
            make_final(21)


if __name__ == "__main__":
    unittest.main()
