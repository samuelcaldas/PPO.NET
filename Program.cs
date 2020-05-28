using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using NumSharp;
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
using Tensorflow;
using Gym.Spaces;

namespace ppo.net
{
    class Program
    {
        #region Python Variables
        // Python Engine Lock handler
        private static IntPtr gs;

        //  Configurações   //
        private static readonly int     TRAINING_EPOCHS = 700;  // Quantidade total de episódios
        private static readonly int     TRAINING_STEPS = 200;   // Quantas sequencias vão acontecer dentro de cada episódio
        private static readonly double  GAMMA = (double)0.9;                   // Avanço (?)
        private static readonly int     NUMBER_OF_SAMPLES = 64; // Tamanho do pacote à entrar para treinamento em cada etapa (?)
        
        // Cria arrais para armazenar os dados dos episódios:
        private static List<dynamic>    buffered_states     = new List<dynamic>();     // buffered_states: buffer do estado
        private static List<dynamic>    buffered_actions    = new List<dynamic>();    // buffered_actions: buffer da ação
        private static List<double>     all_epochs_rewards  = new List<double>();    // all_epochs_rewards: recompensa de todos os episódios
        private static List<double>     buffered_rewards    = new List<double>();      // buffered_rewards: buffer da recompensa         
        private static List<double>     discounted_reward   = new List<double>();     // Cria um array pra armazenar as recompensas calculadas
        //
        private static dynamic  state;
        private static double   epoch_reward;
        private static dynamic  action;
        private static dynamic  step 
        {
            get => (step_state, reward);
            set 
            {
                step_state = value.GetItem(0);  // Adiciona ao buffer de estado o estado atual state
                reward = (double)value.GetItem(1);
            }
        }
        private static dynamic  step_state;
        private static double   reward;
        private static double    value_state_;

        private static double[] _State = new double[3];
        public static dynamic State
        {
            get => _State;
            set
            {
                _State[0] = value.GetItem(0);
                _State[1] = value.GetItem(1);
                _State[2] = value.GetItem(2);
            }
        }

        #endregion
        static void Main(string[] args)
        {
            PythonEngine.Initialize(); // Inicialize o mecanismo Python

            // Adquirindo trava
            gs = PythonEngine.AcquireLock(); // Adquire o bloqueio do interpretador

            //  Importações //
            dynamic site    = Py.Import("site");
            dynamic sys     = Py.Import("sys");
            site.addsitedir(@"C:\Users\samue\source\repos\PPO.NET");
            dynamic _gym     = Py.Import("gym");
            dynamic np   = Py.Import("numpy");   // Numpy para trabalhar com arrays
            dynamic _ppo    = Py.Import("PPO");
            dynamic plt     = Py.Import("matplotlib.pyplot");
            var ppo = _ppo.PPO();                       // Instancia action classe PPO

            PythonEngine.ReleaseLock(gs);

            // Implementação do ambiente   //
            //env = _gym.make("Pendulum-v0").unwrapped;// Instancia o ambiente pendulo
            //var env = gym.Make("Pendulum-v0");
            CartPoleEnv env = new CartPoleEnv(WinFormEnvViewer.Factory);


            Console.Clear();

            try
            {
                //  Loop de episódios //
                for (int EPOCH = 0; EPOCH < TRAINING_EPOCHS; EPOCH++)   // TRAINING_EPOCHS: quantidade de episódios 
                {
                    state = env.Reset();                                // Redefine o ambiente e armazena o estado atual em state

                    epoch_reward = 0;   // Recompensa do episódio
                                        //  Loop de episódio //
                    for (int STEP = 0; STEP < TRAINING_STEPS; STEP++)   // Duração de cada episodio
                    {
                        env.Render();                                   // Renderiza o ambiente
                        gs = PythonEngine.AcquireLock();
                        action = ppo.choose_action(state);              // Envia um estado state e recebe uma ação action 
                        PythonEngine.ReleaseLock(gs);

                        step = env.Step(action);                        // Envia uma ação action ao ambiente e recebe o estado step_state, e action recompensa reward

                        buffered_states.Add(state);             // Adiciona ao buffer de estado o estado atual state

                        buffered_actions.Add(action);           // Adiciona ao buffer de ação action ação atual action
                        buffered_rewards.Add((reward + 8) / 8);  // Adiciona ao buffer de recompensa action recompensa atual (?) normalizada (reward+8)/8

                        state = step_state;                     // Atualiza action variável de estado com o estado recebido pelo ambiente
                        epoch_reward += reward;                 // soma action recompensa da ação action recompensa do episodio

                        //  Atualiza PPO //
                        if ((STEP + 1) % NUMBER_OF_SAMPLES == 0 || STEP == TRAINING_STEPS - 1) // A cada 64 passos a rede é atualizada
                        {
                            gs = PythonEngine.AcquireLock();
                            value_state_ = ppo.get_value(step_state);    // Passa o estado atual step_state e recebe o valor atual da taxa de aprendizagem da CRITICA
                            PythonEngine.ReleaseLock(gs);
                                                                         // V = learned state-value function
                            discounted_reward.Clear();                   // Limpa o array de recompensas calculadas
                            buffered_rewards.Reverse();                  // Coloca o array de buffer de recompensa ao contrario

                            foreach (double buffered_reward in buffered_rewards)
                            {
                                value_state_ = buffered_reward + GAMMA * value_state_;  // Calcula action recompensa multiplicando action recompensa recebida reward pela GAMMA 
                                                                                        // e pelo valor da taxa de aprendizado do estado value_state_
                                discounted_reward.Add(value_state_);        // Adiciona ao array de recompensas calculadas 
                            }
                            discounted_reward.Reverse();                    // Coloca o array de recompensas calculadas ao contrario
                                                                            // vstack transforma os arrays que estão em linha, em colunas
                                                                            // Esses arrays de colunas são armazenados em _buffered_states _buffered_actions e _buffered_rewards
                            var _buffered_states = np.vstack(buffered_states);
                            var _buffered_actions = np.vstack(buffered_actions);
                            var _buffered_rewards = np.expand_dims(np.array(discounted_reward), axis: 1);

                            // Esvazia os buffers de estado, ação e recompensa
                            buffered_states.Clear();
                            buffered_actions.Clear();
                            buffered_rewards.Clear();

                            // Treine o cliente e o ator (status, ações, desconto de reward)
                            gs = PythonEngine.AcquireLock();
                            ppo.update_network(     // Atualiza as redes com:
                                _buffered_states,   //   Os estados acumulados
                                _buffered_actions,  //   As ações acumuladas
                                _buffered_rewards   //   As recompensas acumuladas
                            );
                            PythonEngine.ReleaseLock(gs);
                        }
                    }
                    // Adiciona action recompensa do episodio atual ao array de recompensas
                    if (EPOCH == 0)
                    {
                        all_epochs_rewards.Add(epoch_reward);
                    }
                    else
                    {
                        all_epochs_rewards.Add(all_epochs_rewards.Last() * GAMMA + epoch_reward * 0.1);
                    }
                    //Escreve na tela
                    Console.WriteLine(
                        "Recompensa do episódio " + EPOCH + // Numero do episodio
                        ": " + epoch_reward                 // Recompensa do episodio
                    );
                }
                gs = PythonEngine.AcquireLock();
                plt.plot(   // Plota o gráfico de todas as recompensas
                    np.arange(
                        all_epochs_rewards.Count
                    ),
                    all_epochs_rewards
                );
                plt.xlabel("Episódio");
                plt.ylabel("Média móvel da recompensa de cada época");
                plt.show();
                PythonEngine.ReleaseLock(gs);
            }
            //catch (PythonException error)
            catch (Exception error)
            {
                // Comunique erros com exceções no script python -  isso funciona muito bem com pythonnet.
                Console.Clear();
                Console.WriteLine(error);
            }

            PythonEngine.ReleaseLock(gs);
            // Trava liberada

            PythonEngine.Shutdown(); // Liberando recursos

            Console.ReadKey();
        }
    }
}
