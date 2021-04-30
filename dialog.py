# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import pdb

import numpy as np

from metric import MetricsContainer
import data
import utils
import domain


class DialogLogger(object):
    CODE2ITEM = [
        ('item0', 'book'),
        ('item1', 'hat'),
        ('item2', 'ball'),
    ]

    def __init__(self, verbose=False, log_file=None, append=False):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def dump_ctx(self, name, ctx):
        assert len(ctx) == 6, 'we expect 3 objects'
        s = ' '.join(['%s=(count:%s value:%s)' % (self.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1]) \
            for i in range(3)])
        self._dump_with_name(name, s)

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_choice(self, name, choice):
        def rep(w):
            p = w.split('=')
            if len(p) == 2:
                for k, v in self.CODE2ITEM:
                    if p[0] == k:
                        return '%s=%s' % (v, p[1])
            return w

        self._dump_with_name(name, ' '.join([rep(c) for c in choice]))

    def dump_agreement(self, agree):
        self._dump('Agreement!' if agree else 'Disagreement?!')

    def dump_reward(self, name, agree, reward):
        if agree:
            self._dump_with_name(name, '%d points' % reward)
        else:
            self._dump_with_name(name, '0 points, (potential %d)' % reward)

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])

    def dump_reward(self, name, agree, reward):
        pass


class Dialog(object):
    def __init__(self, agents, args):
        # For now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_moving_percentage('moving_agree')
        self.metrics.register_average('advantage')
        self.metrics.register_moving_average('moving_advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        self.metrics.register_average('agree_comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_moving_average('%s_moving_rew' % agent.name)
            self.metrics.register_average('agree_%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        if self.args.ref_text:
            ref_text = ' '.join(data.read_lines(self.args.ref_text))
            self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and (out[0] in ['<selection>', '<no_agreement>'])

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger, max_words=5000):
        assert len(self.agents) == len(ctxs)
        for agent, ctx, partner_ctx in zip(self.agents, ctxs, reversed(ctxs)):
            agent.feed_context(ctx)
            agent.feed_partner_context(partner_ctx)
            logger.dump_ctx(agent.name, ctx)
        logger.dump('-' * 80)

        # Choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        self.metrics.reset()

        #words_left = np.random.randint(50, 200)
        words_left = max_words
        length = 0
        expired = False

        while True:
            out = writer.write(max_words=words_left)
            words_left -= len(out)
            length += len(out)

            self.metrics.record('sent_len', len(out))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            conv.append(out)
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)

            if self._is_selection(out):
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break

            if words_left <= 1:
                break

            writer, reader = reader, writer


        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)
            logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])

        agree, rewards = self.domain.score_choices(choices, ctxs)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            logger.dump_reward(agent.name, agree, reward)
            j = 1 if i == 0 else 0
            agent.update(agree, reward, choice=choices[i],
                partner_choice=choices[j], partner_input=ctxs[j], max_partner_reward=rewards[j])

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for agent, reward in zip(self.agents, rewards):
                self.metrics.record('agree_%s_rew' % agent.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards

class TripleDialog(object):
    def __init__(self, agents, args):
        # For now we only suppport dialog of 2 agents
        assert len(agents) == 3
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_moving_percentage('moving_agree')
        self.metrics.register_average('advantage')
        self.metrics.register_moving_average('moving_advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        self.metrics.register_average('agree_comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_moving_average('%s_moving_rew' % agent.name)
            self.metrics.register_average('agree_%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        if self.args.ref_text:
            ref_text = ' '.join(data.read_lines(self.args.ref_text))
            self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and (out[0] in ['<selection>', '<no_agreement>'])

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger, max_words=5000):
        assert len(self.agents) == len(ctxs)
        i = 0
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)
            partner_ctx = ctxs[(i + 1) % len(ctxs)]
            agent.feed_partner_context(partner_ctx)
            logger.dump_ctx(agent.name, ctx)
            i += 1
        logger.dump('-' * 80)

        # determine turns
        duels = [[self.agents[i], self.agents[(i+1) % len(self.agents)]]
                 for i in range(len(self.agents))]
        completed = []
        def duel():
            k = 0
            invert = False
            while True:
                a, b = duels[k]
                k = (k + 1) % len(self.agents)
                if (a.name, b.name) not in completed and (b.name, a.name) not in completed:
                    if invert:
                        yield b, a
                    else:
                        yield a, b
                if k % len(self.agents) == 0:
                    invert = not invert

        repeat = 5
        c_repeat = 5
        conv = [[] for _ in range(repeat)]
        self.metrics.reset()
        words_left = max_words
        length = 0
        expired = False
        get_duel = duel()
        done = 0

        
        while c_repeat > 0:
            reader, writer = next(get_duel)
            logger.dump(f'{writer.name} vs {reader.name}')
            out = writer.write(max_words=words_left)
            words_left -= len(out)
            length += len(out)

            self.metrics.record('sent_len', len(out))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            conv[repeat - c_repeat].append(out)
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)

            if self._is_selection(out):
                done += 1
                logger.dump(f'{writer.name} and {reader.name} are done!')
                completed.append((writer.name, reader.name))
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                if done == len(duels):
                    c_repeat -= 1
                    done = 0
                    completed = []

            if words_left < 2:
                c_repeat -= 1
                done = 0
                completed = []

            # input()


        for agent in self.agents:
            agent.model.train()
            agent.sel_model.train()

        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)
            logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])

        agree, rewards = self.domain.score_choices(choices, ctxs)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        print('logprobs:', [len(a.logprobs) for a in self.agents])
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            logger.dump_reward(agent.name, agree, reward)
            print(agent.name, agree, reward)
            j = (i + 1) % len(self.agents)
            if self.args.lola:
                agent.update_lola(agree,
                                  reward,
                                  self.agents[:j] + self.agents[j+1:],
                                  choice=choice[i],
                                  partner_choices=choices[:j] + choices[j+1:],
                                  partner_inputs=ctxs[:j] + ctxs[j+1:],
                                  partner_rewards=rewards[:j] + rewards[j+1:])
            else:
                agent.update(agree, 
                             reward,
                             choice=choices[i],
                             partner_choice=choices[j],
                             partner_input=ctxs[j],
                             partner_reward=rewards[j])

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for agent, reward in zip(self.agents, rewards):
                self.metrics.record('agree_%s_rew' % agent.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', sum([len(c) for c in conv]))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards

class TwoVsOneDialog(object):
    def __init__(self, buyers, seller, args):
        # For now we only suppport dialog of 2 agents
        self.buyers = buyers
        self.seller = seller
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_moving_percentage('moving_agree')
        self.metrics.register_average('advantage')
        self.metrics.register_moving_average('moving_advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        self.metrics.register_average('agree_comb_rew')
        for agent in self.buyers + [self.seller]:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_moving_average('%s_moving_rew' % agent.name)
            self.metrics.register_average('agree_%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        if self.args.ref_text:
            ref_text = ' '.join(data.read_lines(self.args.ref_text))
            self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and (out[0] in ['<selection>', '<no_agreement>'])

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger, max_words=5000):
        assert len(self.buyers) + 1 == len(ctxs)

        # initialize contexts
        self.seller.feed_context(ctxs[0])
        logger.dump_ctx(self.seller.name, ctxs[0])
        for agent, ctx in zip(self.buyers, ctxs[1:]):
            agent.feed_context(ctx)
            agent.feed_partner_context(ctxs[0])
            self.seller.feed_partner_context(ctx)
            logger.dump_ctx(agent.name, ctx)
        logger.dump('-' * 80)

        # determine turns
        completed = []
        def duel():
            k = 0
            invert = False
            while True:
                a, b = self.buyers[k], self.seller
                if (a.name, b.name) not in completed and (b.name, a.name) not in completed:
                    if invert:
                        yield b, a, k
                    else:
                        yield a, b, k
                k = (k + 1) % len(self.buyers)
                if k % len(self.buyers) == 0:
                    invert = not invert

        conv = [[] for _ in self.buyers]
        self.metrics.reset()
        words_left = max_words
        length = 0
        expired = False
        get_duel = duel()
        done = 0

        while True:
            writer, reader, k = next(get_duel)
            logger.dump(f'{writer.name} vs {reader.name}')
            out = writer.write(max_words=words_left)
            words_left -= len(out)
            length += len(out)

            self.metrics.record('sent_len', len(out))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            conv[k].append(out)
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)
                logger.dump('\n')

            if self._is_selection(out):
                done += 1
                logger.dump(f'{writer.name} and {reader.name} are done!')
                completed.append((writer.name, reader.name))
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                if done == len(self.buyers):
                    break

            if words_left <= 1:
                break

        for agent in [self.seller] + self.buyers:
            agent.model.train()
            agent.sel_model.train()

        choices = []
        for agent in [self.seller] + self.buyers:
            choice = agent.choose()
            choices.append(choice)
            logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])

        rewards = self.domain.score_choices_2_vs_1(choices, ctxs)
        agree1 = '<selection>' in conv[0][-1]
        agree2 = '<selection>' in conv[1][-1]
        agree  = agree1 and agree2
        # logger.dump(conv)
        rewards1 = self.domain.score_choices_2_vs_1(choices[::2], ctxs[::2])
        rewards2 = self.domain.score_choices_2_vs_1(choices[:2], ctxs[:2])
        logger.dump('{} vs {}'.format(self.seller.name, self.buyers[1].name))
        logger.dump(rewards1)
        logger.dump_agreement(agree1)
        logger.dump('-' * 80)
        logger.dump('{} vs {}'.format(self.seller.name, self.buyers[0].name))
        logger.dump(rewards2)
        logger.dump_agreement(agree2)
        logger.dump('-' * 80)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        logger.dump(rewards)
        for i, (agent, reward) in enumerate(zip([self.seller] + self.buyers, rewards)):
            logger.dump_reward(agent.name, agree, reward)
            if i > 0:
                seller_reward = None if i > 1 else rewards[0]
                if i % 2 == 0:
                    agent.update_lola(agree, 
                                      reward,
                                      [self.buyers[(i + 1) % len(self.buyers)]],
                                      choice=choices[i],
                                      partner_choices=[choices[(i + 1) % len(self.buyers)]],
                                      partner_inputs=[ctxs[(i + 1) % len(self.buyers)]],
                                      partner_rewards=[rewards[(i + 1) % len(self.buyers)]],
                                      seller_reward=seller_reward)
                else:
                    agent.update(agree,
                                 reward,
                                 choice=choices[i],
                                 partner_choice=choices[(i + 1) % len(self.buyers)],
                                 partner_input=ctxs[(i + 1) % len(self.buyers)],
                                 partner_reward=rewards[(i + 1) % len(self.buyers)],
                                 seller_reward=seller_reward)

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for buyer, reward in zip(self.buyers, rewards):
                self.metrics.record('agree_%s_rew' % buyer.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', sum([len(c) for c in conv]))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip([self.seller] + self.buyers, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards
