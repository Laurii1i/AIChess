import chess
import chess.engine
import torch
import torch.optim as optim
from model2 import ChessNet
from pathlib import Path
import utils as ut
from mappings import two_way
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import torch
from multiprocessing import Process, Queue
from multiprocessing.queues import Empty
import torch.nn.functional as F
import multiprocessing as mp
import time

root = Path(__file__).resolve().parent

def run_selfplay_instance(instance_id, games_per_instance, request_q, output_q, myqueue):

    """
    Each process runs games_per_instance games with its own SelfPlay instance.
    Returns lists of white and black losses for all games it played.
    """
    selfplay = SelfPlay(instance_id, myqueue)
    selfplay.request_queue = request_q

    all_white_states, all_black_states = [], []
    all_white_moves, all_black_moves = [], []
    all_white_rewards, all_black_rewards = [], []
    all_white_masks, all_black_masks = [], []


    try:
        for _ in range(games_per_instance):
            w_states, b_states, w_moves, b_moves, w_rewards, b_rewards, w_masks, b_masks = selfplay.play()

            all_white_states.append(w_states)
            all_black_states.append(b_states)
            all_white_moves.append(w_moves)
            all_black_moves.append(b_moves)
            all_white_rewards.append(w_rewards)
            all_black_rewards.append(b_rewards)
            all_white_masks.append(w_masks)
            all_black_masks.append(b_masks)
            
    finally:
        try:
            selfplay.engine.quit()
            request_q.put(('DONE', instance_id))
        except Exception as e:
            print(f"Engine cleanup failed in process {instance_id}: {e}")

    output_q.put((all_white_states, all_black_states, all_white_moves, all_black_moves, all_white_rewards, all_black_rewards, all_white_masks, all_black_masks))



def compute_total_loss_from_results(results, model, device='cuda'):
    """
    Given results from multiple self-play processes, compute the total policy gradient loss.
    Assumes each result is:
        (white_states, black_states, white_moves, black_moves, 
         white_rewards, black_rewards, white_masks, black_masks)
    """
    
    # Flatten all moves, states, rewards, and masks across processes and games
    flat_white_states = [torch.from_numpy(s) for process in results for game in process[0] for s in game]
    flat_black_states = [torch.from_numpy(s) for process in results for game in process[1] for s in game]

    flat_white_moves = [m for process in results for game in process[2] for m in game]
    flat_black_moves = [m for process in results for game in process[3] for m in game]

    flat_white_rewards = [r for process in results for game in process[4] for r in game]
    flat_black_rewards = [r for process in results for game in process[5] for r in game]

    flat_white_masks = [torch.from_numpy(mask) for process in results for game in process[6] for mask in game]
    flat_black_masks = [torch.from_numpy(mask) for process in results for game in process[7] for mask in game]

    # Convert to tensors on device
    white_states_tensor = torch.stack(flat_white_states).to(device)
    black_states_tensor = torch.stack(flat_black_states).to(device)

    white_actions_tensor = torch.tensor(flat_white_moves, device=device)
    black_actions_tensor = torch.tensor(flat_black_moves, device=device)

    white_rewards_tensor = torch.tensor(flat_white_rewards, device=device, dtype=torch.float)
    black_rewards_tensor = torch.tensor(flat_black_rewards, device=device, dtype=torch.float)

    white_masks_tensor = torch.stack(flat_white_masks).to(device)
    black_masks_tensor = torch.stack(flat_black_masks).to(device)

    # Forward pass
    white_logits = model(white_states_tensor)[0]
    black_logits = model(black_states_tensor)[0]

    # Apply masks (set illegal moves to -inf before softmax)

    white_logits = white_logits.masked_fill(white_masks_tensor == 0, float('-inf'))
    black_logits = black_logits.masked_fill(black_masks_tensor == 0, float('-inf'))

    # Compute log_probs
    white_log_probs = torch.log_softmax(white_logits, dim=-1)
    black_log_probs = torch.log_softmax(black_logits, dim=-1)

    selected_white_log_probs = white_log_probs[range(len(white_actions_tensor)), white_actions_tensor]
    selected_black_log_probs = black_log_probs[range(len(black_actions_tensor)), black_actions_tensor]

    # Compute total policy gradient loss
    white_loss = -(selected_white_log_probs * white_rewards_tensor).sum()
    black_loss = -(selected_black_log_probs * black_rewards_tensor).sum()
    total_loss = white_loss + black_loss

    return total_loss

def gpu_worker(model, device, num_processes, request_q, response_qs, timeout=0.05):
    """
    GPU worker that batches requests from CPU workers, performs forward pass,
    computes log probabilities, and sends results back to the per-worker queues.

    Each request should be a tuple: (worker_id, input_tensor, return_queue)
    """
    model.to(device).train()
    active_workers = set(range(num_processes))
    pending = []
    fag = False
    while active_workers:
        # Try to get new requests
        try:
            item = request_q.get(timeout=timeout)
        except Empty:
            item = None
            continue

        if item[0] == "DONE":
            worker_id = item[1]
            active_workers.remove(worker_id)
            print(f'{worker_id} has finished')
            if not active_workers: # Tyhjä lista
                break
            
        elif item is not None:
            
            pending.append(item)


        if len(pending) == len(active_workers):

            pending.sort(key = lambda x: x[0])
            worker_ids = sorted(list(active_workers))

            tensors =  [x[1] for x in pending]
            batch = torch.cat(tensors, dim=0).to(device, non_blocking=True)

            # Forward pass + log_softmax on GPU
            with torch.no_grad(), torch.inference_mode():
                logits = model(batch)[0]
                log_probs = F.log_softmax(logits, dim=-1).cpu()

            # Send results to each worker's return queue
            for id, probs in zip(worker_ids, log_probs):
                response_qs[id].put(probs)

            pending.clear()

class SelfPlay():

    def __init__(self, id, myqueue, initial_fen: str | None = None):
        
        root = Path(__file__).resolve().parent
        self.board = chess.Board(initial_fen) if initial_fen else chess.Board()
        self.id = id
        self.engine = chess.engine.SimpleEngine.popen_uci(root / 'Stockfish' / 'stockfish-windows-x86-64-avx2.exe')
        self.engine.configure({"Threads": 2})
        self.model_weights_file = "improved_model1_groups_0-14.pth"
        self.model = self._load_model(self.model_weights_file)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4, eps=1e-8)
        self.model.train()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = myqueue
        


    def _load_model(self, model_file: str):
        root = Path(__file__).resolve().parent
        model = ChessNet()
        model.load_state_dict(torch.load(root / model_file))
        return model
    
    def play(self):

        white_losses, black_losses = [], []

        #print(f'Starting game {i}')
        white_rewards, black_rewards, white_states, black_states, white_moves, black_moves, white_moves_mask, black_moves_mask = [], [], [], [], [], [], [], []
        self.white_is_mating, self.black_is_mating = False, False
        pov_score_before = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))['score']

        while not self.board.is_game_over():

            fen = self.board.fen()
            legal_moves_mask = ut.get_mask(fen)
            input_tensor = ut.fen_to_tensor(fen)

            white_moves_mask.append(legal_moves_mask.numpy()) if self.board.turn == chess.WHITE else black_moves_mask.append(legal_moves_mask.numpy())

            white_states.append(input_tensor.numpy()) if self.board.turn == chess.WHITE else black_states.append(input_tensor.numpy())

            move_id = self._make_move(input_tensor, legal_moves_mask)

            white_moves.append(move_id) if self.board.turn == chess.BLACK else black_moves.append(move_id)

            """print('----------------')
            if self.board.turn == chess.BLACK:
                print(f'White played {two_way[move_id]}')
            else:
                print(f'Black played {two_way[move_id]}')"""

            pov_score_after = self.engine.analyse(self.board, chess.engine.Limit(time=0.01))['score']
            
            reward = self.score_move(pov_score_before, pov_score_after)

            """if reward == 1:
                print('The move was classified as good\n')
            elif reward == 0:
                print('The move was classified as neutral\n')
            else:
                print('The move was classified as BAD\n')

            print(self.board)"""


            white_rewards.append(reward) if self.board.turn == chess.BLACK else black_rewards.append(reward)

            pov_score_before = pov_score_after

        white_rewards, black_rewards = self.scale_rewards_according_to_game_outcome(white_rewards, black_rewards)   
        white_rewards, black_rewards = self.normalize(white_rewards), self.normalize(black_rewards)
        self.board.reset()
        return white_states, black_states, white_moves, black_moves, white_rewards, black_rewards, white_moves_mask, black_moves_mask     
        

        white_losses.append(-(torch.stack(white_log_probs) * white_rewards).sum())
        black_losses.append(-(torch.stack(black_log_probs) * black_rewards).sum())

        


        if i % batch_size == 0 and i != 0:

            self._back_propagate(torch.stack(white_losses), torch.stack(black_losses))
            white_losses, black_losses, white_rewards, black_rewards, white_log_probs, black_log_probs = [], [], [], [], [], []
            self.save_weights()
            print('Weights updated')

    def scale_rewards_according_to_game_outcome(self, white_rewards, black_rewards):

        if self.board.result() == '1-0': #Valkonen voitti

            white_rewards = [0.2 + i * 1.2 if i >= 0 else i for i in white_rewards]
            black_rewards = [-0.2 + i * 1.2 if i >= 0 else i for i in black_rewards]

            if len(white_rewards) < 70:
                white_rewards = [i + 0.3 for i in white_rewards]
            if len(white_rewards) < 35:
                white_rewards = [i + 0.5 for i in white_rewards]
            #print('White won, final configuration: \n')
            #print(self.board)
        
        elif self.board.result() == '0-1':
            black_rewards = [0.2 + i * 1.2 if i >= 0 else i for i in black_rewards]
            white_rewards = [-0.2 + i * 1.2 if i >= 0 else i for i in white_rewards]

            if len(black_rewards) < 70:
                black_rewards = [i + 0.3 for i in black_rewards]
            if len(black_rewards) < 35:
                black_rewards = [i + 0.5 for i in black_rewards]
            #print('Black won, final configuration: \n')
            #print(self.board)

        return white_rewards, black_rewards

    def score_move(self, score_before, score_after):

        good_move_score, medium_move_score, bad_move_score = 1, 0, -1
        if not score_before.is_mate(): # Jos stockfish ei nähnyt forcettu mattia ennen siirtoa

            if score_after.is_mate(): # Nyt on blunderoitu

                self.white_is_mating = True if self.board.turn == chess.WHITE else False # musta blunderoi, nyt valkoinen voi yrittää löytää matin
                self.black_is_mating = True if self.board.turn == chess.BLACK else False
                return bad_move_score
            
            else:

                score_diff = np.abs(score_after.pov(chess.WHITE).score() - score_before.pov(chess.WHITE).score()) # katsotaan miten evaluaatio (centipawn score) on muuttunut

                if score_diff < 80: return good_move_score
                elif 80 <= score_diff < 150: return medium_move_score
                else: return bad_move_score

        else:

            if not score_after.is_mate(): # Blunderoitu ulos mattichainista
                if self.white_is_mating:

                    self.white_is_mating = False
                    if score_after.pov(chess.WHITE).score() < 0:
                        return bad_move_score
                    else: return medium_move_score
                else:
                    self.black_is_mating = False
                    if score_after.pov(chess.WHITE).score() > 0:
                        return bad_move_score
                    else: return medium_move_score

            else: # Matitus jatkuu (tai siirtynyt toiselle osapuolelle)
                mate_sequence_length_before = score_before.pov(chess.WHITE).mate()
                mate_sequence_length_after = score_after.pov(chess.WHITE).mate()

                if np.abs(mate_sequence_length_before) > np.abs(mate_sequence_length_after): # Nyt on tehty oikea siirto force mattia kohti

                    # Tarkistetaan vielä, oliko siirto "jatkumoa" ketjulle, vai turpaan ottavan osapuolen pakotettu siirto

                    if (self.white_is_mating and self.board.turn == chess.BLACK) or (self.black_is_mating and self.board.turn == chess.WHITE):
                        return good_move_score
                    else:
                        return medium_move_score
                
                elif mate_sequence_length_before == mate_sequence_length_after:
                    return medium_move_score
                
                elif (mate_sequence_length_after * mate_sequence_length_before) < 0: # nyt on blunderoitu ja annettu toiselle mahdollisuus matittaa

                    if self.board.turn == chess.BLACK:
                        self.white_is_mating, self.black_is_mating = False, True
                        return bad_move_score 
                    else:
                        self.white_is_mating, self.black_is_mating = True, False
                        return bad_move_score
                else: # backwards in mate sequence
                    return medium_move_score

    def _back_propagate(self, white_losses, black_losses):

        loss = white_losses.sum() + black_losses.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_final_result(self):

        result = self.board.result()
        if result == '1-0':
            return 1.0   # White wins
        elif result == '0-1':
            return -1.0  # Black wins
        else:
            return 0.0   # Draw

    def _store_move(self, move_id):
        if self.board.turn == chess.WHITE:
            self.white_moves.append(move_id)                
        else:
            self.black_moves.append(move_id) 

    def _make_move(self, input_tensor, legal_moves_mask, training=True):
        """
        Sends input_tensor to GPU worker via request_q and waits for log probabilities
        on the per-worker queue.
        """

        # Ensure batch dimension for concatenation in GPU worker
        input_tensor = input_tensor.unsqueeze(0)

        # Send request to GPU worker along with this worker's return queue      
        self.request_queue.put((self.id, input_tensor))

        # Wait for result from GPU worker
        log_probs = self.queue.get()       

        # Mask illegal moves
        log_probs = log_probs.clone() # ota ehkä pois
        log_probs[legal_moves_mask == 0] = float('-inf')

        # Convert log_probs to probabilities
        probs = torch.exp(log_probs)

        # Choose move
        if training:
            move_idx = torch.distributions.Categorical(probs).sample()
        else:
            move_idx = torch.argmax(probs)

        # Play the move
        uci = two_way[move_idx.item()]
        self.board.push_uci(uci)

        return move_idx.item()

    def get_score(self):

        a = self.engine.analyse(self.board, chess.engine.Limit(depth=0.1))['score']
        return a
    
    def normalize(self, x):
        """
        Normalize a list of floats to zero mean and unit variance.

        Args:
            x: list of floats

        Returns:
            list of floats (normalized)
        """
        if not x:
            return []
        
        mean = sum(x) / len(x)
        variance = sum((xi - mean) ** 2 for xi in x) / len(x)
        std = variance ** 0.5 if variance > 0 else 1e-8
        normalized = [(xi - mean) / std for xi in x]
        return normalized

    
    def save_weights(self):
        torch.save(self.model.state_dict(), self.model_weights_file)
        pass

    
    def __del__(self):
        pass
        #self.engine.quit()
if __name__ == '__main__':

    num_processes = 4
    games_per_process = 6  # 16*64 = 1024 games total
   
    device = 'cuda'
    loop_count = 0
    #mp.set_start_method("spawn") 

    manager = mp.Manager()
    
    request_q = manager.Queue()
    output_q = manager.Queue()
    response_queues = {i: manager.Queue() for i in range(num_processes)}

    selfplay = SelfPlay(id = None, myqueue = None)
    selfplay.model.to('cuda')
    
    
    while True:

        loop_count += 1
        all_white_losses, all_black_losses = [], []

        gpu_proc = Process(target=gpu_worker, args=(selfplay.model, "cuda:0", num_processes, request_q, response_queues))
        gpu_proc.start()
        processes = []
        for i in range(num_processes):
            p = Process(target=run_selfplay_instance,
                        args=(i, games_per_process, request_q, output_q, response_queues[i]))
            p.start()
            processes.append(p)



        all_results = []
        for _ in range(num_processes):
            results = output_q.get()  # blocking until available
            all_results.append(results)
        loss = compute_total_loss_from_results(all_results, model = selfplay.model)

        for p in processes:
            p.join()
        
        
        
        selfplay.optimizer.zero_grad()
        loss.backward()
        selfplay.optimizer.step()
        selfplay.save_weights()

        torch.cuda.empty_cache()
        del all_results
        print(f'{num_processes * games_per_process * loop_count} games ran and weigths saved')

