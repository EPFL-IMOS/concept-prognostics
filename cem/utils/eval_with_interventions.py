import torch
import numpy as np

# from data.ncmapss import scale_concept

import warnings
warnings.simplefilter("ignore", UserWarning)


def eval_with_interventions(model, dataloader, cycles, n_concepts, window_size=50, stride=1, theta_df=None):
    fault_count = torch.zeros(n_concepts)
    fault_count_per_cycle = torch.zeros(n_concepts)
    intervene = [False] * n_concepts
    intervene_done = [False] * n_concepts
    rul = []
    intervene_ind = [[] for _ in range(n_concepts)]
    batch_size = dataloader.batch_size
    previous_cycle = 0
    n_batches = 0
    cem = hasattr(model, "pre_concept_model")

    for idx, batch in enumerate(dataloader):
        n_batches += 1
        x, y, c = batch
        x = x.to(device=model.device)
        cycle_batch = cycles[idx * batch_size * stride:min(len(cycles), (idx+1) * batch_size * stride)]
        first_cycle = cycle_batch[0]
        last_cycle = cycle_batch[-1]
        # theta = theta_df.iloc[idx * batch_size * stride]

        # get embedding
        if cem:  # CEM
            pre_c = model.pre_concept_model(x)

            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(model.concept_context_generators):
                if model.shared_prob_gen:
                    prob_gen = model.concept_prob_generators[0]
                else:
                    prob_gen = model.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(model.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
        else: # CBM
            latent = model.x2c_model(x)
            if model.extra_dims:
                # Then we only sigmoid on the probability bits but
                # let the other entries up for grabs
                c_pred_probs = model.sig(latent[:, :-model.extra_dims])
                c_others = model.bottleneck_nonlin(latent[:,-model.extra_dims:])
                # c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
                c_sem = c_pred_probs
            else:
                c_pred = model.sig(latent)
                c_sem = c_pred

        # c_sem = probability
        # for con in range(n_concepts):

            # if (c_sem[0][con].detach().cpu().numpy() > 0.5):
            #     # check of it is correct or not:
            #     if (c[0][con].detach().cpu().numpy() < 0.5):
            #         fault_count[con] +=1
            #     else:
            #         fault_count[con] = 0

        probs = torch.zeros_like(c).to(device=model.device)

        # STRATEGY 1: intervene when a fault is activated for DELAY cycles (average per batch)
        # DELAY = 1
        # fault_count_per_cycle += c_sem.mean(axis=0).cpu()  # mean of concept activations over batch
        # if last_cycle != previous_cycle:  # batch is going into next cycle
        #     for con in range(n_concepts):
        #         if fault_count_per_cycle[con] / n_batches > 0.5:
        #             fault_count[con] += 1
        #         if fault_count[con] > DELAY: # intervene at next cycle
        #             intervene[con] = True
        #             print(f"Intervention needed on {con} at cycle {first_cycle}/{last_cycle} (total: {cycles[-1]})")
        #         fault_count_per_cycle[con] = 0
        #     previous_cycle = last_cycle
        #     n_batches = 0
        # for con in range(n_concepts):
        #     if intervene_done[con] or intervene[con] or any(intervene[c] and not intervene_done[c] for c in range(n_concepts) if c != con):
        #         if not intervene_done[con] and c[:,con].max() == 0:  # it's actually healthy, stop intervention
        #             intervene[con] = False
        #             print(f"Healthy, stop intervention on {con}.")
        #             probs[:,con] = c_sem[:,con]
        #         elif not intervene_done[con] and first_cycle != last_cycle:
        #             inter_idx_start = np.argwhere(cycle_batch == last_cycle)[0][0]
        #             probs[inter_idx_start:,con] = 1
        #             probs[:inter_idx_start,con] = c_sem[:inter_idx_start,con]
        #             intervene[con] = True
        #             print(f"Intervened on {con}!")
        #         else:
        #             inter_idx_start = 0
        #             probs[:,con] = 1
        #             intervene[con] = True
        #             print(f"Intervened on {con}!")
        #     else:
        #         probs[:,con] = c_sem[:,con]

        # for con in range(n_concepts):
        #     if intervene[con] and not intervene_done[con]:
        #         intervene_ind[con] += [0] * inter_idx_start + [1] * (batch_size - inter_idx_start)
        #         intervene_done[con] = True
        #         print(f"Intervention done for {con}")
        #     else:
        #         intervene_ind[con] += [int(intervene_done[con])] * batch_size

        # STRATEGY 2: intervene when a fault is activated for DELAY cycles (average per batch) but only intervene on the faulty component
        DELAY = 1
        fault_count_per_cycle += c_sem.mean(axis=0).cpu()  # mean of concept activations over batch
        if last_cycle != previous_cycle:  # batch is going into next cycle
            for con in range(n_concepts):
                if fault_count_per_cycle[con] / n_batches > 0.5:
                    fault_count[con] += 1
                if fault_count[con] > DELAY: # intervene at next cycle
                    intervene[con] = True
                    print(f"Intervention needed on {con} at cycle {first_cycle}/{last_cycle} (total: {cycles[-1]})")
                fault_count_per_cycle[con] = 0
            previous_cycle = last_cycle
            n_batches = 0
        for con in range(n_concepts):
            if intervene_done[con] or intervene[con]:
                if not intervene_done[con] and c[:,con].max() == 0:  # it's actually healthy, stop intervention
                    intervene[con] = False
                    print(f"Healthy, stop intervention on {con}.")
                    probs[:,con] = c_sem[:,con]
                elif not intervene_done[con] and first_cycle != last_cycle:
                    inter_idx_start = np.argwhere(cycle_batch == last_cycle)[0][0]
                    probs[inter_idx_start:,con] = 1 #c[inter_idx_start:,con]
                    probs[:inter_idx_start,con] = c_sem[:inter_idx_start,con]
                    print(f"Intervened on {con}!")
                else:
                    inter_idx_start = 0
                    probs[:,con] = 1 # c[:,con]
                    print(f"Intervened on {con}!")
            else:
                probs[:,con] = c_sem[:,con]

        for con in range(n_concepts):
            if intervene[con] and not intervene_done[con]:
                intervene_ind[con] += [0] * inter_idx_start + [1] * (batch_size - inter_idx_start)
                intervene_done[con] = True
                print(f"Intervention done for {con}")
            else:
                intervene_ind[con] += [int(intervene_done[con])] * batch_size

        # STRATEGY 3: intervene on each component at a fixed inspection schedule, every PERIOD cycles
        # PERIOD = 20
        # first_intervention = True
        # for con in range(n_concepts):
        #     if cycle % PERIOD == 0 or intervene[con]:
        #         if first_intervention:
        #             intervene_ind += [1] * batch_size
        #             first_intervention = False
        #         intervene[con] = True
        #         probs[:,con] = c[:,con]
        #         if c[:,con].max() == 0:  # it's actually healthy, stop intervention
        #             intervene[con] = False
        #     else:
        #         if not any(intervene) and con == (n_concepts - 1):  # last concept
        #             intervene_ind += [0] * batch_size
        #         probs[:,con] = c_sem[:,con]

        # STRATEGY oracle: always intervene
        # first_intervention = True
        # for con in range(n_concepts):
        #     if first_intervention:
        #         intervene_ind += [1] * batch_size
        #         first_intervention = False
        #     intervene[con] = True
        #     probs[:,con] = c[:,con]

        # STRATEGY continuous oracle: always intervene and use continuous theta values
        # first_intervention = True
        # for con in range(n_concepts):
        #     if first_intervention:
        #         intervene_ind += [1] * batch_size
        #         first_intervention = False
        #     intervene[con] = True
        #     probs[:,con] = scale_concept(theta[con])

        if cem:
            c_pred = (
                contexts[:, :, :model.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, model.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
            )
            c_pred = c_pred.view((-1, model.emb_size * model.n_concepts))
        else:
            if model.extra_dims:
                c_pred = torch.cat([probs, c_others], axis=-1)
            else:
                c_pred = probs

        if not cem and model.bool:
            y = model.c2y_model((c_pred > 0.5).float())
        else:
            y = model.c2y_model(c_pred)
        rul.append(y[:,0].cpu().detach())

    rul = torch.cat(rul, axis=0)
    return rul, np.array([intervene_ind[con][:len(rul)] for con in range(n_concepts)])
