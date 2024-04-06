//
//  ViewController.swift
//  CV
//
//  Created by Liberty Hamilton on 3/14/18.
//  Copyright © 2018 Liberty Hamilton. All rights reserved.
//

import UIKit
import AVFoundation
import GameplayKit
import CoreAudio


class ViewController: UIViewController, AVAudioPlayerDelegate {
    
    var stringPassed:String = "1"
    var stimSet:Int = 0 // Default is tasks (not short)
    
    @IBOutlet var greenCircle: UIImageView!
    
    @IBOutlet weak var startButton: UIButton!
    @IBOutlet weak var cvLabel: UILabel!
    
    // This is the progress bar that shows how many trials have elapsed
    @IBOutlet weak var cvProgress: UIProgressView!
    
    // Link to the audio player
    var cvAudioPlayer:AVAudioPlayer!
    var currentSound:String!
    
    // Get audio recorder
//    var recorder: AVAudioRecorder!
//    var levelTimer = Timer()
//    let LEVEL_THRESHOLD: Float = -20.0
//    var isLoud: Bool = false
    
    // List of files and number of trials
    var fileList = [String]() // This will be a list of all audio files to play
    var i = 0 // This is the counter for which file to play
    var t = 0 // This is the counter for the number of trials
    var numTrials = 1
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let stimulusSetList = ["tasks", "tasks_short"]
        let filename = stimulusSetList[stimSet]
        print("The chosen stimulus set is")
        print(filename)
        
        if let path = Bundle.main.path(forResource: filename, ofType: "txt") {
            do {
                let text = try String(contentsOfFile: path, encoding: String.Encoding.utf8)
                print(text)
                fileList = text.components(separatedBy: "\n")
                fileList = fileList.filter({ $0 != ""}) // Get rid of blanks
            } catch {
                print("Failed to read text from \(filename)")
            }
        } else {
            print("Failed to load file from app bundle \(filename)")
        }
        
        numTrials = fileList.count
        print("number of trials is:")
        print(numTrials)
    }
    
    
    override func viewWillAppear(_ animated: Bool) {
        // Put the balloons off screen at the bottom
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func playAudio(_ sender: UIButton) {
        greenCircle.isHidden = true
        cvLabel.isHidden = false
        if t < numTrials {
            if i == fileList.count {
                i = 0
            }
            print(fileList[i])
            let voiceSound = NSURL(fileURLWithPath: Bundle.main.path(forResource: "click", ofType: "wav")!)
            currentSound = voiceSound.deletingPathExtension?.lastPathComponent
            
            startButton.setTitle("Next", for: .normal)
            do {
                cvAudioPlayer = try AVAudioPlayer(contentsOf: voiceSound as URL)
                cvAudioPlayer.prepareToPlay()
                cvAudioPlayer.enableRate = true
                cvAudioPlayer.rate = 1.0
            
                //cvAudioPlayer.numberOfLoops = -1
            } catch {
                print("Problem in getting File")
            }
            cvAudioPlayer.delegate = self
        
            cvAudioPlayer.play()
            cvLabel.text = fileList[i]
            
            i+=1 // file count
            t+=1 // trial count
            cvProgress.progress+=1.0/Float32(numTrials)
        }
        else { // At the end of having presented all of the sounds, show the "Great job!" message
            cvLabel.text = "Great job!"
            startButton.isHidden = true
            
            //recorder.stop()
            
            i = 0 // reset the file number
            t = 0 // reset the trial number
            cvProgress.progress = 0.0
            greenCircle.isHidden = true
            
            // Allow the user to restart the whole thing
            UIView.animate(withDuration: 1.1, delay: 2.2, animations: {
            self.startButton.setTitle("Start over", for: .normal)
            self.startButton.isHidden = false
            })

        }
    }
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool){
        if flag == true{
            sleep(UInt32(0.75)) // Wait 0.75 seconds before showing green circle
            greenCircle.isHidden = false
            startButton.isHidden = true
            let seconds = 3.0
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { [self] in
                // Put your code which should be executed with a delay here
                self.greenCircle.isHidden = true
                startButton.isHidden = false
                cvLabel.isHidden = true
            }
            
        }
    }

}


