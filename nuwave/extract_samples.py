import os

speakers = ['p257', 'p232']
numbers = [366, 84, 43, 304, 374, 20, 22, 167, 294, 49,274, 345, 291, 144, 275, 342, 305, 4, 279, 278, 383, 93, 60, 165, 85, 163, 171, 76, 21, 219, 203, 232, 210, 146, 14, 130, 16, 89, 18, 175]
path_result = '/home/nlp/wolhanr/NuWave/nuwave/test_sample/result'

dir = os.listdir(path_result)
new_dir = 'to_annotate'
print(len(os.listdir(new_dir)))
for speaker in speakers:
    for number in numbers:
        try:
            if len(str(number)) == 1:
                os.rename(os.path.join(path_result,f'test_{speaker}_00{number}.wav_up.wav'), os.path.join(new_dir, f'{speaker}_00{number}.wav'))
            if len(str(number)) == 2:
                os.rename(os.path.join(path_result,f'test_{speaker}_0{number}.wav_up.wav'), os.path.join(new_dir, f'{speaker}_0{number}.wav'))
            if len(str(number)) == 3:
                os.rename(os.path.join(path_result,f'test_{speaker}_{number}.wav_up.wav'), os.path.join(new_dir, f'{speaker}_{number}.wav'))
        except:
            continue


